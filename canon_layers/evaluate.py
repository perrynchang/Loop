"""
Evaluation script for trained Canon Layers models.
Computes task-specific accuracy metrics as described in the paper.
"""
import argparse
import random
import torch
import torch.nn.functional as F
from collections import defaultdict

from models import build_transformer, build_loop_transformer
from tasks.depo import DepoTokenizer
from tasks.brevo import BrevoTokenizer, build_random_dag, topological_reachable
from tasks.mano import ManoTokenizer, build_expr, eval_expr, serialize_expr, MOD
from tasks.lano import LanoTokenizer, CFG_RULES, CFG_ROOTS, generate_sentence, is_valid_cfg
from tasks.bios import (
    BioSDataset, BioSTokenizer, ATTR_NAMES,
    BioS32Dataset, BioS32Tokenizer,
    CLASSIFY_TYPES, COMPARE_TYPES, INVERSE_TYPES,
    _parse_augment,
)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt['args']

    from tasks import (DepoTokenizer, BrevoTokenizer, ManoTokenizer, LanoTokenizer)
    task_vocab = {
        'depo': DepoTokenizer(args.get('variant', 'depo1')).total_vocab,
        'brevo': BrevoTokenizer(args.get('variant', 'brevo1')).total_vocab,
        'mano': ManoTokenizer().total_vocab,
        'lano': LanoTokenizer().total_vocab,
        'capo': 256,
        'bios':   BioSTokenizer(N=args.get('N', 1000)).total_vocab,
        'bios32': BioS32Tokenizer(N=args.get('N', 1000)).total_vocab,
    }
    vocab_size = task_vocab.get(args['task'], 256)

    use_loop = args.get('model_type', 'transformer') == 'loop'
    if use_loop:
        model = build_loop_transformer(
            vocab_size=vocab_size,
            size=args['model_size'],
            rope=(args['rope'] != 'none'),
            rope_fraction=args.get('rope_fraction', 1.0),
            canon_positions=args.get('canon', ''),
            canon_residual=args.get('canon_residual', True),
            max_seq_len=2048,
            T_max=args.get('T_max', 4),
        )
    else:
        model = build_transformer(
            vocab_size=vocab_size,
            size=args['model_size'],
            rope=(args['rope'] != 'none'),
            rope_fraction=args.get('rope_fraction', 1.0),
            canon_positions=args.get('canon', ''),
            canon_residual=args.get('canon_residual', True),
            max_seq_len=2048,
        )

    state = {k: v for k, v in ckpt['model_state'].items()
             if not any(k.endswith(s) for s in ('attn.mask', 'rope.cos_cached', 'rope.sin_cached'))}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, args


@torch.no_grad()
def evaluate_depo(model, variant="depo1", N=225, K=8, n_samples=200, device='cpu'):
    """
    Evaluate k-hop reasoning depth on Depo task.
    Returns dict: {k: accuracy} for k in [1, K].
    """
    tok = DepoTokenizer(variant)
    rng = random.Random(99999)
    results = defaultdict(list)

    for _ in range(n_samples):
        n = N  # evaluate on hardest case
        nodes = list(range(n))
        rng.shuffle(nodes)
        perm = {nodes[i]: nodes[(i + 1) % n] for i in range(n)}

        for k in range(1, K + 1):
            q = rng.choice(nodes)
            cur = q
            for _ in range(k):
                cur = perm[cur]

            # Build input sequence
            edges = list(perm.items())
            rng.shuffle(edges)
            tokens = [tok.BOS]
            for x, y in edges:
                tokens += tok.encode_node(x)
                tokens += tok.encode_node(y)
            tokens.append(tok.query_token(k))
            tokens += tok.encode_node(q)

            x_in = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model(x_in)
            pred = logits[0, -1].argmax().item()

            # Expected: first token of answer node encoding
            expected = tok.encode_node(cur)[0]
            results[k].append(int(pred == expected))

    return {k: sum(v) / len(v) for k, v in results.items()}


@torch.no_grad()
def evaluate_brevo(model, variant="brevo1", N=70, n_samples=100, device='cpu'):
    """Evaluate DAG traversal accuracy on Brevo task."""
    tok = BrevoTokenizer(variant)
    rng = random.Random(88888)
    correct = 0
    total = 0

    for _ in range(n_samples):
        n = N
        edges = build_random_dag(n, max_degree=4, rng=rng)
        children = defaultdict(list)
        for u, v in edges:
            children[u].append(v)

        nodes_with_children = [u for u in range(n) if children[u]]
        if not nodes_with_children:
            continue
        q = rng.choice(nodes_with_children)
        expected_answer = topological_reachable(q, children, n)
        if not expected_answer:
            continue

        # Build input: edges + query
        tokens = [tok.BOS]
        edge_list = list(edges)
        rng.shuffle(edge_list)
        for u, v in edge_list:
            tokens += tok.encode_node(u, rng)
            tokens += tok.encode_node(v, rng)
        tokens.append(tok.QUERY)
        tokens += tok.encode_node(q, rng)
        tokens.append(tok.ANS)

        # Generate answer tokens autoregressively
        x_in = torch.tensor([tokens], dtype=torch.long, device=device)
        generated = []
        for _ in range(len(expected_answer) * tok.node_max_len + 5):
            logits = model(x_in)
            next_tok = logits[0, -1].argmax().item()
            if next_tok == tok.EOS:
                break
            generated.append(next_tok)
            x_in = torch.cat([x_in, torch.tensor([[next_tok]], device=device)], dim=1)

        # Compare generated to expected (simplified: check length)
        expected_tokens = []
        for a in expected_answer:
            expected_tokens += tok.encode_node(a, rng)

        correct += int(generated == expected_tokens)
        total += 1

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_mano(model, L=10, n_samples=500, device='cpu'):
    """Evaluate modular arithmetic accuracy at max length L."""
    tok = ManoTokenizer()
    rng = random.Random(77777)
    correct = 0

    for _ in range(n_samples):
        tree = build_expr(L, rng)
        expected = eval_expr(tree) % MOD
        expr_toks = serialize_expr(tree)

        prefix = [tok.BOS, tok.len_token(L)] + tok.encode_expr(expr_toks) + [tok.ANS]
        x_in = torch.tensor([prefix], dtype=torch.long, device=device)
        logits = model(x_in)
        pred = logits[0, -1].argmax().item()
        correct += int(pred == tok.val_token(expected))

    return correct / n_samples


@torch.no_grad()
def evaluate_lano(model, variant="cfg3f", n_samples=200, max_gen_len=500, device='cpu'):
    """Evaluate CFG sentence validity."""
    tok = LanoTokenizer()
    rules = CFG_RULES[variant]
    root = CFG_ROOTS[variant]
    correct = 0

    for _ in range(n_samples):
        x_in = torch.tensor([[tok.BOS]], dtype=torch.long, device=device)
        generated = []
        for _ in range(max_gen_len):
            logits = model(x_in)
            probs = torch.softmax(logits[0, -1], dim=-1)
            next_tok = torch.multinomial(probs, 1).item()
            if next_tok == tok.BOS:
                break
            generated.append(next_tok)
            x_in = torch.cat([x_in, torch.tensor([[next_tok]], device=device)], dim=1)

        if generated and is_valid_cfg(generated, rules, root):
            correct += 1

    return correct / n_samples


@torch.no_grad()
def evaluate_bios32_classify(model, N=1000, augment="permute", n_samples=500,
                              with_cot=False, device='cpu'):
    """
    Part 3.2 — classification accuracy for each query type.

    with_cot=False: pure answer prediction (no hint; this is the hard case).
    with_cot=True:  feed the correct attribute value hint before COT_SEP and
                    check the token immediately after it.

    Returns dict {query_type: accuracy, 'mean': mean}.
    """
    ds  = BioS32Dataset(N=N, augment=augment, cot_prob=0.0)
    tok = ds.tok
    rng = random.Random(55555)
    results = defaultdict(list)
    do_permute, do_fullname, _, _ = _parse_augment(augment)

    for _ in range(n_samples):
        pid   = rng.randrange(N)
        attrs = ds.persons[pid]
        p_tok = tok.person_tok(pid)

        # Biography context
        blocks = [(tok.bio_mark(name), tok.attr_tok(name, attrs[name])) for name in ATTR_NAMES]
        if do_permute:
            rng.shuffle(blocks)
        bio = [tok.BOS, p_tok]
        for mark, value in blocks:
            if do_fullname:
                bio.append(p_tok)
            bio.append(mark)
            bio.append(value)

        for q_type in CLASSIFY_TYPES:
            if q_type == 'month_even':
                q_mark   = tok.Q_MONTH_EVEN
                hint_tok = tok.attr_tok('month', attrs['month'])
                expected = tok.YES if attrs['month'] % 2 == 0 else tok.NO
            elif q_type == 'month_mod6':
                q_mark   = tok.Q_MONTH_MOD6
                hint_tok = tok.attr_tok('month', attrs['month'])
                expected = tok.mod6_tok(attrs['month'])
            elif q_type == 'major_lucky':
                q_mark   = tok.Q_MAJOR_LUCKY
                hint_tok = tok.attr_tok('major', attrs['major'])
                expected = tok.lucky_tok(attrs['major'])
            elif q_type == 'major_lucky_mod5':
                q_mark   = tok.Q_MAJOR_LUCKY_MOD5
                hint_tok = tok.attr_tok('major', attrs['major'])
                expected = tok.mod5_tok(attrs['major'])

            prefix = bio + [tok.QUERY, q_mark, p_tok]
            if with_cot:
                prefix += [hint_tok, tok.COT_SEP]

            x_in   = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(x_in)
            pred   = logits[0, -1].argmax().item()
            results[q_type].append(int(pred == expected))

    per_type = {name: (sum(v) / len(v) if v else 0.0) for name, v in results.items()}
    all_vals = [v for vals in results.values() for v in vals]
    per_type['mean'] = sum(all_vals) / len(all_vals) if all_vals else 0.0
    return per_type


@torch.no_grad()
def evaluate_bios32_compare(model, N=1000, augment="permute", n_samples=500,
                             with_cot=False, device='cpu'):
    """
    Part 3.2 — comparison accuracy for each query type.

    Both persons' bios are provided as context. with_cot=True prepends the
    relevant attribute values (for A and B) before COT_SEP.

    Returns dict {query_type: accuracy, 'mean': mean}.
    """
    ds  = BioS32Dataset(N=N, augment=augment, cot_prob=0.0)
    tok = ds.tok
    rng = random.Random(44444)
    results = defaultdict(list)
    do_permute, do_fullname, _, _ = _parse_augment(augment)

    def _bio(pid):
        attrs  = ds.persons[pid]
        p_tok  = tok.person_tok(pid)
        blocks = [(tok.bio_mark(name), tok.attr_tok(name, attrs[name])) for name in ATTR_NAMES]
        if do_permute:
            rng.shuffle(blocks)
        t = [tok.BOS, p_tok]
        for mark, value in blocks:
            if do_fullname:
                t.append(p_tok)
            t.append(mark)
            t.append(value)
        return t

    for _ in range(n_samples):
        pid_a = rng.randrange(N)
        pid_b = rng.randrange(N)
        while pid_b == pid_a:
            pid_b = rng.randrange(N)

        attrs_a = ds.persons[pid_a]
        attrs_b = ds.persons[pid_b]
        pa, pb  = tok.person_tok(pid_a), tok.person_tok(pid_b)
        context = _bio(pid_a) + _bio(pid_b)

        for q_type in COMPARE_TYPES:
            if q_type == 'month_rank':
                q_mark    = tok.Q_MONTH_RANK
                hint_toks = [tok.attr_tok('month', attrs_a['month']),
                             tok.attr_tok('month', attrs_b['month'])]
                expected  = tok.YES if attrs_a['month'] > attrs_b['month'] else tok.NO
            elif q_type == 'month_diff':
                q_mark    = tok.Q_MONTH_DIFF
                hint_toks = [tok.attr_tok('month', attrs_a['month']),
                             tok.attr_tok('month', attrs_b['month'])]
                expected  = tok.month_diff_tok(attrs_a['month'] - attrs_b['month'])
            elif q_type == 'major_lucky_rank':
                q_mark    = tok.Q_MAJOR_LUCKY_RANK
                hint_toks = [tok.lucky_tok(attrs_a['major']),
                             tok.lucky_tok(attrs_b['major'])]
                la = tok.major_luckiness[attrs_a['major']]
                lb = tok.major_luckiness[attrs_b['major']]
                expected  = tok.YES if la > lb else tok.NO
            elif q_type == 'major_lucky_diff':
                q_mark    = tok.Q_MAJOR_LUCKY_DIFF
                hint_toks = [tok.lucky_tok(attrs_a['major']),
                             tok.lucky_tok(attrs_b['major'])]
                diff      = tok.major_luckiness[attrs_a['major']] - tok.major_luckiness[attrs_b['major']]
                expected  = tok.major_diff_tok(diff)

            prefix = context + [tok.QUERY, q_mark, pa, pb]
            if with_cot:
                prefix += hint_toks + [tok.COT_SEP]

            x_in   = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(x_in)
            pred   = logits[0, -1].argmax().item()
            results[q_type].append(int(pred == expected))

    per_type = {name: (sum(v) / len(v) if v else 0.0) for name, v in results.items()}
    all_vals = [v for vals in results.values() for v in vals]
    per_type['mean'] = sum(all_vals) / len(all_vals) if all_vals else 0.0
    return per_type


@torch.no_grad()
def evaluate_bios32_inverse(model, N=1000, augment="permute+multi5+reverse6",
                             n_samples=500, device='cpu'):
    """
    Part 3.2 — inverse search accuracy for each query type.

    The bio is provided in the same reverse format used during training.
    Without 'reverse<N>' in augment this should return ~0% (the paper's main
    negative result). With reverse6 it can work once the model is trained on
    reversed biographies.

    Returns dict {query_type: accuracy, 'mean': mean}.
    """
    ds  = BioS32Dataset(N=N, augment=augment, cot_prob=0.0)
    tok = ds.tok
    rng = random.Random(33333)
    results = defaultdict(list)
    do_permute, _, _, reverse_pos = _parse_augment(augment)

    for _ in range(n_samples):
        pid   = rng.randrange(N)
        attrs = ds.persons[pid]
        p_tok = tok.person_tok(pid)

        blocks = [(tok.bio_mark(name), tok.attr_tok(name, attrs[name])) for name in ATTR_NAMES]
        if do_permute:
            rng.shuffle(blocks)

        bio = [tok.BOS]
        if reverse_pos == 0:
            bio.append(p_tok)
        for i, (mark, value) in enumerate(blocks):
            bio.append(mark)
            bio.append(value)
            if reverse_pos > 0 and i + 1 == reverse_pos:
                bio.append(p_tok)
        if reverse_pos >= len(blocks):
            bio.append(p_tok)

        for q_type in INVERSE_TYPES:
            attr_name = q_type[4:]
            attr_val  = attrs[attr_name]
            Q_MARK = {
                'inv_month': tok.Q_INV_MONTH,
                'inv_city':  tok.Q_INV_CITY,
                'inv_univ':  tok.Q_INV_UNIV,
            }
            prefix   = bio + [tok.QUERY, Q_MARK[q_type], tok.attr_tok(attr_name, attr_val)]
            expected = p_tok

            x_in   = torch.tensor([prefix], dtype=torch.long, device=device)
            logits = model(x_in)
            pred   = logits[0, -1].argmax().item()
            results[q_type].append(int(pred == expected))

    per_type = {name: (sum(v) / len(v) if v else 0.0) for name, v in results.items()}
    all_vals = [v for vals in results.values() for v in vals]
    per_type['mean'] = sum(all_vals) / len(all_vals) if all_vals else 0.0
    return per_type


@torch.no_grad()
def evaluate_bios(model, N=1000, augment="", n_samples=500, device='cpu'):
    """
    Evaluate per-attribute QA accuracy on the BioS knowledge extraction task.

    Presents a full biography (with the specified augmentation) then queries
    [QUERY][Q_MARK][PERSON] and checks whether the model's argmax equals the
    correct attribute value token. Mirrors the QA generation accuracy metric
    from Figure 3 of "Physics of Language Models: Part 3.1".

    Returns dict: {attr_name: accuracy, ..., 'mean': mean_accuracy}
    """
    ds  = BioSDataset(N=N, augment=augment)
    tok = ds.tok
    rng = random.Random(55555)
    results = defaultdict(list)

    for _ in range(n_samples):
        pid   = rng.randrange(N)
        attrs = ds.persons[pid]
        p_tok = tok.person_tok(pid)

        # Build bio tokens (same augmentation as training, but no QA appended)
        blocks = [(tok.bio_mark(name), tok.attr_tok(name, attrs[name]))
                  for name in ATTR_NAMES]
        if 'permute' in augment:
            rng.shuffle(blocks)

        tokens = [tok.BOS, p_tok]
        for mark, value in blocks:
            if 'fullname' in augment:
                tokens.append(p_tok)
            tokens.append(mark)
            tokens.append(value)

        # Query one random attribute
        q_name = rng.choice(ATTR_NAMES)
        tokens += [tok.QUERY, tok.query_mark(q_name), p_tok]

        x_in   = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = model(x_in)
        pred   = logits[0, -1].argmax().item()
        expected = tok.attr_tok(q_name, attrs[q_name])
        results[q_name].append(int(pred == expected))

    per_attr = {name: (sum(v) / len(v) if v else 0.0) for name, v in results.items()}
    all_vals = [v for vals in results.values() for v in vals]
    per_attr['mean'] = sum(all_vals) / len(all_vals) if all_vals else 0.0
    return per_attr


def main():
    p = argparse.ArgumentParser(description="Evaluate Canon Layers models")
    p.add_argument("checkpoint", help="Path to model checkpoint")
    p.add_argument("--task", choices=["depo", "brevo", "mano", "lano", "bios", "bios32"])
    p.add_argument("--variant", default="")
    p.add_argument("--N", type=int, default=225)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--L", type=int, default=10)
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--with_cot", action="store_true", default=False,
                   help="bios32: evaluate with correct CoT hint prepended")
    default_device = "mps" if torch.backends.mps.is_available() else "cpu"
    p.add_argument("--device", default=default_device)
    args = p.parse_args()

    device = torch.device(args.device)
    model, train_args = load_model(args.checkpoint, device)
    task = args.task or train_args['task']
    variant = args.variant or train_args.get('variant', '')

    print(f"Evaluating {task} ({variant}) ...")

    if task == "depo":
        results = evaluate_depo(model, variant, args.N, args.K, args.n_samples, device)
        for k, acc in sorted(results.items()):
            print(f"  k={k:2d}: {acc*100:.1f}%")
    elif task == "brevo":
        acc = evaluate_brevo(model, variant, args.N, args.n_samples, device)
        print(f"  Accuracy: {acc*100:.1f}%")
    elif task == "mano":
        acc = evaluate_mano(model, args.L, args.n_samples, device)
        print(f"  Accuracy (L={args.L}): {acc*100:.1f}%")
    elif task == "lano":
        acc = evaluate_lano(model, variant or "cfg3f", args.n_samples, device=device)
        print(f"  CFG validity: {acc*100:.1f}%")
    elif task == "bios":
        results = evaluate_bios(model, N=args.N, augment=variant, n_samples=args.n_samples, device=device)
        mean = results.pop('mean')
        for attr, acc in results.items():
            print(f"  {attr:>8s}: {acc*100:.1f}%")
        print(f"  {'mean':>8s}: {mean*100:.1f}%")
    elif task == "bios32":
        N = args.N if args.N != 225 else 1000  # use sensible default for bios
        cot_label = " (with CoT hint)" if args.with_cot else " (no CoT)"

        print(f"\n--- Classification{cot_label} ---")
        res = evaluate_bios32_classify(model, N=N, augment=variant,
                                       n_samples=args.n_samples, with_cot=args.with_cot, device=device)
        mean = res.pop('mean')
        for qt, acc in res.items():
            print(f"  {qt:>20s}: {acc*100:.1f}%")
        print(f"  {'mean':>20s}: {mean*100:.1f}%")

        print(f"\n--- Comparison{cot_label} ---")
        res = evaluate_bios32_compare(model, N=N, augment=variant,
                                      n_samples=args.n_samples, with_cot=args.with_cot, device=device)
        mean = res.pop('mean')
        for qt, acc in res.items():
            print(f"  {qt:>20s}: {acc*100:.1f}%")
        print(f"  {'mean':>20s}: {mean*100:.1f}%")

        print(f"\n--- Inverse Search ---")
        res = evaluate_bios32_inverse(model, N=N, augment=variant,
                                      n_samples=args.n_samples, device=device)
        mean = res.pop('mean')
        for qt, acc in res.items():
            print(f"  {qt:>20s}: {acc*100:.1f}%")
        print(f"  {'mean':>20s}: {mean*100:.1f}%")


if __name__ == "__main__":
    main()
