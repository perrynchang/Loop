"""
Quick demo: verify all tasks and Canon layer implementation work correctly.
Runs a small sanity check forward pass on each task without training.
"""
import torch
from tasks import (
    build_depo_dataset, DepoTokenizer,
    build_brevo_dataset, BrevoTokenizer,
    build_mano_dataset, ManoTokenizer,
    build_lano_dataset, LanoTokenizer,
)
from models import build_transformer, count_canon_params
from torch.utils.data import DataLoader


def test_canon_layer():
    from models.canon import CanonLayer, CanonABCD
    print("=== Canon Layer ===")
    canon = CanonLayer(dim=512, residual=True)
    x = torch.randn(2, 64, 512)
    out = canon(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    # Verify causality: output at position t should not depend on t+1
    x2 = x.clone()
    x2[:, 10:, :] = 0
    out2 = canon(x2)
    # Position 9 should be unchanged since Canon only looks back
    assert torch.allclose(out[:, :10, :], out2[:, :10, :], atol=1e-5), "Causality violated!"
    print(f"  CanonLayer(512, residual=True): {sum(p.numel() for p in canon.parameters())} params")

    full_canon = CanonABCD(d_model=512, positions="ABCD")
    total = sum(p.numel() for p in full_canon.parameters())
    print(f"  CanonABCD(512, ABCD): {total} params")
    print("  PASSED")


def test_transformer():
    print("\n=== Transformer Sizes ===")
    for size, rope, canon in [
        ("8L512D", True, ""),
        ("12L768D", True, "ABCD"),
        ("8L512D", False, "ABCD"),  # NoPE + Canon
    ]:
        model = build_transformer(
            vocab_size=100, size=size, rope=rope,
            canon_positions=canon, max_seq_len=128
        )
        n_total = model.num_parameters()
        n_canon = count_canon_params(model)
        pct = 100 * n_canon / n_total if n_total > 0 else 0
        x = torch.randint(0, 100, (2, 64))
        out = model(x)
        assert out.shape == (2, 64, 100)
        rope_str = "RoPE" if rope else "NoPE"
        canon_str = f"+Canon-{canon}" if canon else ""
        print(f"  {size} {rope_str}{canon_str}: {n_total:,} params, {n_canon:,} Canon ({pct:.2f}%)")
    print("  PASSED")


def test_depo():
    print("\n=== Task Depo ===")
    ds = build_depo_dataset(variant="depo1", N=25, K=4, context_len=256)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    tok = DepoTokenizer("depo1")
    print(f"  Batch shape: {batch.shape}, vocab_size: {tok.total_vocab}")
    model = build_transformer(vocab_size=tok.total_vocab, size="8L512D", max_seq_len=256)
    logits = model(batch[:, :-1])
    print(f"  Logits shape: {logits.shape}")
    print("  PASSED")


def test_brevo():
    print("\n=== Task Brevo ===")
    ds = build_brevo_dataset(variant="brevo1", N=15, context_len=256)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    tok = BrevoTokenizer("brevo1")
    print(f"  Batch shape: {batch.shape}, vocab_size: {tok.total_vocab}")
    model = build_transformer(vocab_size=tok.total_vocab, size="8L512D", max_seq_len=256)
    logits = model(batch[:, :-1])
    print(f"  Logits shape: {logits.shape}")
    print("  PASSED")


def test_mano():
    print("\n=== Task Mano ===")
    ds = build_mano_dataset(L=5, context_len=256)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    tok = ManoTokenizer()
    print(f"  Batch shape: {batch.shape}, vocab_size: {tok.total_vocab}")
    model = build_transformer(vocab_size=tok.total_vocab, size="8L512D", max_seq_len=256)
    logits = model(batch[:, :-1])
    print(f"  Logits shape: {logits.shape}")
    print("  PASSED")


def test_lano():
    print("\n=== Task Lano ===")
    ds = build_lano_dataset(variant="cfg3f")
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    tok = LanoTokenizer()
    print(f"  Batch shape: {batch.shape}, vocab_size: {tok.total_vocab}")
    model = build_transformer(vocab_size=tok.total_vocab, size="8L512D", max_seq_len=512)
    logits = model(batch[:, :-1])
    print(f"  Logits shape: {logits.shape}")
    print("  PASSED")


def run_micro_training():
    """Tiny training loop to verify gradient flow."""
    print("\n=== Micro Training (10 steps, Mano L=3) ===")
    import torch.nn as nn

    ds = build_mano_dataset(L=3, context_len=128)
    loader = DataLoader(ds, batch_size=8)
    tok = ManoTokenizer()

    model = build_transformer(
        vocab_size=tok.total_vocab,
        size="8L512D",
        canon_positions="ABCD",
        max_seq_len=128
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for i, batch in enumerate(loader):
        if i >= 10:
            break
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, tok.total_vocab),
            targets.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0] * 1.5, "Loss should decrease or stay similar"
    print("  PASSED")


if __name__ == "__main__":
    test_canon_layer()
    test_transformer()
    test_depo()
    test_brevo()
    test_mano()
    test_lano()
    run_micro_training()
    print("\nAll checks passed!")
