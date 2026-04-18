"""
Training script for Canon Layers experiments.
Supports all five synthetic tasks (Depo, Brevo, Mano, Lano, Capo).
"""
import argparse
import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import build_transformer, count_canon_params
from tasks import (
    build_depo_dataset, DepoTokenizer,
    build_brevo_dataset, BrevoTokenizer,
    build_mano_dataset, ManoTokenizer,
    build_lano_dataset, LanoTokenizer,
    build_capo_dataset,
)


def get_task_config(task, variant, N, K, L):
    """Return (dataset, vocab_size, context_len) for a given task."""
    if task == "depo":
        ds = build_depo_dataset(variant=variant, N=N, K=K)
        tok = DepoTokenizer(variant)
        return ds, tok.total_vocab, 2048
    elif task == "brevo":
        ds = build_brevo_dataset(variant=variant, N=N)
        tok = BrevoTokenizer(variant)
        return ds, tok.total_vocab, 1024 if variant == "brevo1" else 1536
    elif task == "mano":
        ds = build_mano_dataset(L=L)
        tok = ManoTokenizer()
        return ds, tok.total_vocab, 1024
    elif task == "lano":
        ds = build_lano_dataset(variant=variant)
        tok = LanoTokenizer()
        return ds, tok.total_vocab, 512 if variant == "cfg3f" else 1536
    elif task == "capo":
        ds = build_capo_dataset(N=N)
        return ds, 256, 512
    else:
        raise ValueError(f"Unknown task: {task}")


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr_frac=0.1):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return max_lr * min_lr_frac
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return max_lr * (min_lr_frac + 0.5 * (1 - min_lr_frac) * (1 + math.cos(math.pi * progress)))


def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Build dataset
    dataset, vocab_size, context_len = get_task_config(
        args.task, args.variant, args.N, args.K, args.L
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True,
    )

    # Build model
    model = build_transformer(
        vocab_size=vocab_size,
        size=args.model_size,
        rope=(args.rope != "none"),
        rope_fraction=args.rope_fraction,
        canon_positions=args.canon,
        canon_residual=args.canon_residual,
        max_seq_len=context_len,
        tie_weights=args.tie_weights,
    ).to(device)

    n_params = model.num_parameters()
    n_canon = count_canon_params(model)
    print(f"Model parameters: {n_params:,} total, {n_canon:,} Canon ({100*n_canon/n_params:.2f}%)")
    print(f"Architecture: {args.model_size} | RoPE={args.rope} | Canon={args.canon}")

    # Optimizer: AdamW with paper's hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )

    # Training loop
    model.train()
    step = 0
    total_loss = 0.0
    log_interval = 100
    start_time = time.time()

    print(f"\nTraining for {args.max_steps} steps...")
    for batch in dataloader:
        if step >= args.max_steps:
            break

        # Update learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x = batch.to(device)
        # Language modeling: predict next token
        inputs = x[:, :-1]
        targets = x[:, 1:]

        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f"Step {step:6d} | loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
            total_loss = 0.0

    # Save checkpoint
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({
            'step': step,
            'model_state': model.state_dict(),
            'args': vars(args),
        }, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")

    return model


def parse_args():
    p = argparse.ArgumentParser(description="Train Canon Layers models")

    # Task
    p.add_argument("--task", choices=["depo", "brevo", "mano", "lano", "capo"],
                   default="depo", help="Synthetic task")
    p.add_argument("--variant", default="depo1",
                   help="Task variant (e.g. depo1/depo2, brevo1/brevo2, cfg3f/cfg3j/cfg3k)")
    p.add_argument("--N", type=int, default=225, help="Max graph/permutation size")
    p.add_argument("--K", type=int, default=8, help="Max hop depth (Depo)")
    p.add_argument("--L", type=int, default=10, help="Max expression length (Mano)")

    # Model
    p.add_argument("--model_size", default="8L512D",
                   choices=["8L512D", "12L512D", "8L768D", "12L768D"],
                   help="Model size: {layers}L{hidden}D")
    p.add_argument("--rope", choices=["rope", "nope", "none"], default="rope",
                   help="Positional encoding type")
    p.add_argument("--rope_fraction", type=float, default=1.0,
                   help="Fraction of dims to apply RoPE (1.0=full, 0.25=quarter)")
    p.add_argument("--canon", default="", help="Canon positions: ABCD, AC, AbCD, etc.")
    p.add_argument("--canon_residual", action="store_true", default=True,
                   help="Use residual connections in Canon layers")
    p.add_argument("--no_canon_residual", dest="canon_residual", action="store_false")
    p.add_argument("--tie_weights", action="store_true", default=False,
                   help="Tie embedding and output weights (used for Capo)")

    # Training
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.03)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--warmup_steps", type=int, default=1000)

    # Output
    p.add_argument("--save_path", default="", help="Path to save checkpoint")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
