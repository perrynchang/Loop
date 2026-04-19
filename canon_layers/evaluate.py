"""
Evaluation script for trained Canon Layers models.
Computes task-specific accuracy metrics as described in the paper.
"""
import argparse
import random
import torch
import torch.nn.functional as F
from collections import defaultdict

from models import build_transformer
from tasks.depo import DepoTokenizer
from tasks.brevo import BrevoTokenizer, build_random_dag, topological_reachable
from tasks.mano import ManoTokenizer, build_expr, eval_expr, serialize_expr, MOD
from tasks.lano import LanoTokenizer, CFG_RULES, CFG_ROOTS, generate_sentence, is_valid_cfg


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
    }
    vocab_size = task_vocab.get(args['task'], 256)

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


def main():
    p = argparse.ArgumentParser(description="Evaluate Canon Layers models")
    p.add_argument("checkpoint", help="Path to model checkpoint")
    p.add_argument("--task", choices=["depo", "brevo", "mano", "lano"])
    p.add_argument("--variant", default="")
    p.add_argument("--N", type=int, default=225)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--L", type=int, default=10)
    p.add_argument("--n_samples", type=int, default=200)
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


if __name__ == "__main__":
    main()
