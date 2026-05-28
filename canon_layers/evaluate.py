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
from tasks.depo import DepoTokenizer, generate_node_words
from tasks.brevo import BrevoTokenizer, build_random_dag, topological_reachable, _generate_brevo_words
from tasks.mano import ManoTokenizer, build_expr, eval_expr, serialize_expr, MOD
from tasks.lano import LanoTokenizer, CFG_RULES, CFG_ROOTS, generate_sentence, is_valid_cfg
from tasks.bios import (
    BioSDataset, BioSTokenizer, ATTR_NAMES,
    BioS32Dataset, BioS32Tokenizer,
    CLASSIFY_TYPES, COMPARE_TYPES, INVERSE_TYPES,
    _parse_augment,
)
from tasks.capo import (
    _generate_attrs, _generate_text, _get_tokenizer, _get_compact_vocab,
    get_capo_vocab_size, CAPO_VOCAB_SIZE,
    FIRST_NAMES, MIDDLE_NAMES, LAST_NAMES, CITIES,
    EMPLOYERS, UNIVERSITIES, MAJORS, BIRTH_MONTHS, BIRTH_DAYS, BIRTH_YEARS,
)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt['args']

    from tasks import (DepoTokenizer, BrevoTokenizer, ManoTokenizer, LanoTokenizer)
    task_vocab = {
        'depo': DepoTokenizer(args.get('variant', 'depo1')).total_vocab,
        'brevo': BrevoTokenizer(args.get('variant', 'brevo1'), N=args.get('N', 110)).total_vocab,
        'mano': ManoTokenizer().total_vocab,
        'lano': LanoTokenizer().total_vocab,
        'capo': get_capo_vocab_size(),
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
            tie_weights=args.get('tie_weights', False),
        )

    state = {k: v for k, v in ckpt['model_state'].items()
             if not any(k.endswith(s) for s in ('attn.mask', 'rope.cos_cached', 'rope.sin_cached'))}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, args


@torch.no_grad()
def evaluate_depo(model, variant="depo1", N=225, K=8, n_samples=50, device='cpu',
                  context_len=2048):
    """
    Evaluate k-hop reasoning depth on Depo (teacher-forcing, packed windows).

    Matches the paper protocol (Appendix A.1): n=N fixed, instances packed into
    context_len windows, accuracy computed over all <ans>/answer-token positions
    via a single forward pass (teacher-forcing). Evaluated at k=K and k=K//2.

    Returns {k: accuracy} for k in {K//2, K}.
    """
    tok = DepoTokenizer(variant)
    rng = random.Random(99999)
    correct = defaultdict(int)
    total   = defaultdict(int)

    eval_ks = sorted({K // 2, K})

    for target_k in eval_ks:
        for _ in range(n_samples):
            token_buf, mask_buf = [], []

            # Pack complete instances until the buffer covers a full window.
            # Left-alignment: first instance is never truncated.
            while len(token_buf) < context_len:
                word_list = generate_node_words(rng, N, tok)
                nodes = list(range(N))
                rng.shuffle(nodes)
                perm = {nodes[i]: nodes[(i + 1) % N] for i in range(N)}

                edges = list(perm.items())
                rng.shuffle(edges)
                inst_toks = [tok.BOS]
                inst_mask = [0]
                for x, y in edges:
                    xt, yt = word_list[x], word_list[y]
                    inst_toks += xt + yt
                    inst_mask += [0] * (len(xt) + len(yt))

                for q in rng.sample(nodes, min(10, N)):
                    cur = q
                    for _ in range(target_k):
                        cur = perm[cur]
                    q_toks = word_list[q]
                    a_toks = word_list[cur]
                    inst_toks += [tok.query_token(target_k)] + q_toks + [tok.ANS] + a_toks
                    inst_mask += [0] * (1 + len(q_toks)) + [0] + [1] * len(a_toks)

                token_buf.extend(inst_toks)
                mask_buf.extend(inst_mask)

            chunk_toks = token_buf[:context_len]
            chunk_mask = mask_buf[:context_len]

            x = torch.tensor([chunk_toks], dtype=torch.long, device=device)
            logits = model(x)  # (1, context_len, vocab)

            # Teacher-forcing: at each masked position p, logits[p-1] predicts token[p].
            for p in range(1, context_len):
                if chunk_mask[p]:
                    pred = logits[0, p - 1].argmax().item()
                    correct[target_k] += int(pred == chunk_toks[p])
                    total[target_k]   += 1

    return {k: (correct[k] / total[k] if total[k] > 0 else 0.0) for k in eval_ks}


@torch.no_grad()
def evaluate_brevo(model, variant="brevo1", N=70, n_samples=100, device='cpu'):
    """Evaluate DAG traversal accuracy on Brevo task."""
    tok = BrevoTokenizer(variant, N=N)
    rng = random.Random(88888)
    correct = 0
    total = 0

    for _ in range(n_samples):
        n = N
        edges = build_random_dag(n, max_degree=4, rng=rng)
        parents = defaultdict(list)
        for u, v in edges:
            parents[v].append(u)

        start = max(3 * n // 4, n - 1)
        candidates = [v for v in range(start, n) if parents[v]]
        if not candidates:
            continue
        q = rng.choice(candidates)
        expected_answer = topological_reachable(q, parents, rng)
        if not expected_answer:
            continue

        if tok.variant == "brevo2":
            words = _generate_brevo_words(
                rng, n, tok.vocab_size, tok.node_min_len, tok.node_max_len, tok.NODE_OFFSET
            )
            rng.shuffle(words)
            word_map = {node_id: list(words[node_id]) for node_id in range(n)}
            rev_word_map = {tuple(w): node_id for node_id, w in word_map.items()}
            encode = word_map.__getitem__
        else:
            perm = rng.sample(range(tok.vocab_size), n)
            tok_to_node = {tok.NODE_OFFSET + perm[v]: v for v in range(n)}
            encode = lambda node_id: [tok.NODE_OFFSET + perm[node_id]]

        # Build input: edges + query
        tokens = [tok.BOS]
        edge_list = list(edges)
        rng.shuffle(edge_list)
        for u, v in edge_list:
            tokens += encode(u)
            tokens += encode(v)
        tokens.append(tok.QUERY)
        tokens += encode(q)
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

        # Decode generated tokens back to node IDs
        if tok.variant == "brevo2":
            gen_nodes = []
            word_buf = []
            for t in generated:
                word_buf.append(t)
                if tok.NODE_OFFSET + tok.vocab_size < t <= tok.NODE_OFFSET + 2 * tok.vocab_size:
                    node_id = rev_word_map.get(tuple(word_buf))
                    if node_id is not None:
                        gen_nodes.append(node_id)
                    word_buf = []
        else:
            gen_nodes = [tok_to_node[t] for t in generated if t in tok_to_node]

        # Accept any valid topological ordering of the expected ancestor set
        expected_set = set(expected_answer)
        if set(gen_nodes) == expected_set:
            seen = set()
            valid = True
            for node in gen_nodes:
                for parent in parents.get(node, []):
                    if parent in expected_set and parent not in seen:
                        valid = False
                        break
                if not valid:
                    break
                seen.add(node)
            correct += int(valid)
        total += 1

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_mano(model, L=10, n_samples=500, device='cpu', context_len=1024):
    """
    Evaluate accuracy on packed 1024-token context windows.
    Matches the paper's protocol: l=L expressions are packed end-to-end and
    accuracy is computed over all instances including non-first ones (~40/window at L=10).
    """
    tok = ManoTokenizer()
    rng = random.Random(77777)
    correct = 0
    total = 0

    for _ in range(n_samples):
        buffer = []
        ans_targets = []  # (ans_token_position, expected_val_token)

        while len(buffer) < context_len:
            tree = build_expr(L, rng)
            result = eval_expr(tree) % MOD
            expr_toks = serialize_expr(tree)
            instance = ([tok.BOS, tok.len_token(L)]
                        + tok.encode_expr(expr_toks)
                        + [tok.ans_token(L), tok.val_token(result), tok.EOS])
            ans_pos = len(buffer) + len(instance) - 3
            buffer.extend(instance)
            if ans_pos < context_len:
                ans_targets.append((ans_pos, tok.val_token(result)))

        chunk = torch.tensor([buffer[:context_len]], dtype=torch.long, device=device)
        logits = model(chunk)  # (1, context_len, vocab_size)

        for ans_pos, expected in ans_targets:
            pred = logits[0, ans_pos].argmax().item()
            correct += int(pred == expected)
            total += 1

    return correct / total if total > 0 else 0.0


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


@torch.no_grad()
def evaluate_capo(model, checkpoint_args, N_eval=1000, device='cpu'):
    """
    Evaluate Capo knowledge memorization.

    For N_eval people from the training set, generates a held-out paraphrase
    (exposure index 100, unseen during training which used indices 0-99), then:
      1. Computes NLL loss on each bio.
      2. For each of the 6 attribute types, finds the attribute value's first
         token position in the tokenized text and checks whether the model's
         argmax at that position is correct.
      3. Estimates bits-per-parameter (BPP) from per-attribute accuracy.

    Returns a dict with loss, per-attribute accuracy, mean accuracy, and BPP.
    """
    import math
    N    = checkpoint_args.get('N', 50000)
    seed = 42
    tok  = _get_tokenizer()
    gpt2_to_compact, _, compact_vocab_size = _get_compact_vocab()

    # Rebuild training attributes with identical seed used in CapoDataset
    rng_attrs = random.Random(seed)
    all_attrs = [_generate_attrs(rng_attrs, i) for i in range(N)]

    # Sample N_eval random persons
    eval_rng = random.Random(98765)
    eval_pids = eval_rng.sample(range(N), min(N_eval, N))

    losses = []
    attr_correct = defaultdict(int)
    attr_total   = defaultdict(int)

    ATTR_KEYS = ['birthday', 'birthcity', 'university', 'field', 'company1name', 'company1city']

    for pid in eval_pids:
        attrs   = all_attrs[pid]
        # Exposure 100 was never seen during training (training used 0–99)
        exp_rng = random.Random(seed + pid * 100 + 100)
        text    = _generate_text(attrs, exp_rng)

        # Tokenize with character-to-token offset mapping, then remap to compact IDs
        enc     = tok(text, return_offsets_mapping=True)
        gpt2_ids = enc['input_ids']
        offsets  = enc['offset_mapping']
        tokens   = [gpt2_to_compact[t] for t in gpt2_ids]

        if len(tokens) < 2:
            continue

        t = torch.tensor([tokens], dtype=torch.long, device=device)

        # NLL on the full bio
        logits = model(t[:, :-1])
        loss   = F.cross_entropy(logits.reshape(-1, compact_vocab_size), t[:, 1:].reshape(-1))
        losses.append(loss.item())

        # Pre-compute argmax predictions at every position
        preds = logits[0].argmax(dim=-1)  # shape (T-1,)

        # Build attribute value strings exactly as they appear in the bio
        birthday_str = f"{attrs['birthmonth']} {attrs['birthday']}, {attrs['birthyear']}"
        attr_vals = {
            'birthday':     birthday_str,
            'birthcity':    attrs['birthcity'],
            'university':   attrs['university'],
            'field':        attrs['field'],
            'company1name': attrs['company1name'],
            'company1city': attrs['company1city'],
        }

        for attr_key, attr_val in attr_vals.items():
            char_pos = text.find(str(attr_val))
            if char_pos == -1:
                continue

            # Map character position → first token index of the attribute value
            token_pos = next(
                (i for i, (start, _) in enumerate(offsets) if start >= char_pos),
                None,
            )
            if token_pos is None or token_pos == 0 or token_pos >= len(tokens):
                continue

            expected  = tokens[token_pos]
            predicted = preds[token_pos - 1].item()
            attr_correct[attr_key] += int(predicted == expected)
            attr_total[attr_key]   += 1

    mean_loss   = sum(losses) / len(losses) if losses else float('inf')

    per_attr_acc = {k: attr_correct[k] / attr_total[k]
                    for k in ATTR_KEYS if attr_total[k] > 0}
    mean_acc     = sum(per_attr_acc.values()) / len(per_attr_acc) if per_attr_acc else 0.0

    # Theoretical bits per person (all independent attributes)
    bits_per_person = (
        math.log2(len(FIRST_NAMES)) + math.log2(len(MIDDLE_NAMES)) + math.log2(len(LAST_NAMES)) +
        math.log2(len(BIRTH_MONTHS) * len(BIRTH_DAYS) * len(BIRTH_YEARS)) +
        math.log2(len(CITIES)) + math.log2(len(UNIVERSITIES)) +
        math.log2(len(MAJORS))  + math.log2(len(EMPLOYERS))
    )

    # BPP using queried attributes only (birthday, birthcity, university, field, company)
    queried_bits = (
        math.log2(len(BIRTH_MONTHS) * len(BIRTH_DAYS) * len(BIRTH_YEARS)) +
        math.log2(len(CITIES)) + math.log2(len(UNIVERSITIES)) +
        math.log2(len(MAJORS))  + math.log2(len(EMPLOYERS))
    )
    # Deduplicate tied weights (embedding == lm_head) before counting
    seen_ptrs = set()
    n_params = 0
    for p in model.parameters():
        if p.data_ptr() not in seen_ptrs:
            seen_ptrs.add(p.data_ptr())
            n_params += p.numel()
    bpp      = N * queried_bits * mean_acc / n_params

    return {
        'mean_loss':              mean_loss,
        'per_attr_accuracy':      per_attr_acc,
        'mean_accuracy':          mean_acc,
        'bits_per_person':        bits_per_person,
        'queried_bits_per_person': queried_bits,
        'bpp_estimate':           bpp,
        'n_params':               n_params,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate Canon Layers models")
    p.add_argument("checkpoint", help="Path to model checkpoint")
    p.add_argument("--task", choices=["depo", "brevo", "mano", "lano", "capo", "bios", "bios32"])
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
    elif task == "capo":
        results = evaluate_capo(model, train_args, N_eval=args.n_samples, device=device)
        print(f"  Test NLL (held-out paraphrase): {results['mean_loss']:.4f}")
        print(f"  Per-attribute first-token accuracy:")
        for attr, acc in results['per_attr_accuracy'].items():
            print(f"    {attr:>12s}: {acc*100:.1f}%")
        print(f"  Mean accuracy:         {results['mean_accuracy']*100:.1f}%")
        print(f"  Bits/person (theory):  {results['bits_per_person']:.2f} bits")
        print(f"  Queried bits/person:   {results['queried_bits_per_person']:.2f} bits")
        print(f"  Model parameters:      {results['n_params']:,}")
        print(f"  Estimated BPP:         {results['bpp_estimate']:.4f} bits/param")
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
