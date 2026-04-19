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
from tqdm import tqdm

from models import build_transformer, count_canon_params
from tasks import (
    build_depo_dataset, DepoTokenizer,
    build_brevo_dataset, BrevoTokenizer,
    build_mano_dataset, ManoTokenizer,
    build_lano_dataset, LanoTokenizer,
    build_capo_dataset,
)


TASK_DEFAULT_CONTEXT = {
    "depo": 2048, "brevo1": 1024, "brevo2": 1536,
    "mano": 1024, "cfg3f": 512, "cfg3j": 1536, "cfg3k": 1536, "capo": 512,
}


def get_task_config(task, variant, N, K, L, context_len_override=None):
    """Return (dataset, vocab_size, context_len) for a given task."""
    if task == "depo":
        ctx = context_len_override or TASK_DEFAULT_CONTEXT["depo"]
        ds = build_depo_dataset(variant=variant, N=N, K=K, context_len=ctx)
        tok = DepoTokenizer(variant)
        return ds, tok.total_vocab, ctx
    elif task == "brevo":
        ctx = context_len_override or TASK_DEFAULT_CONTEXT.get(variant, 1024)
        ds = build_brevo_dataset(variant=variant, N=N, context_len=ctx)
        tok = BrevoTokenizer(variant)
        return ds, tok.total_vocab, ctx
    elif task == "mano":
        ctx = context_len_override or TASK_DEFAULT_CONTEXT["mano"]
        ds = build_mano_dataset(L=L, context_len=ctx)
        tok = ManoTokenizer()
        return ds, tok.total_vocab, ctx
    elif task == "lano":
        ctx = context_len_override or TASK_DEFAULT_CONTEXT.get(variant, 512)
        ds = build_lano_dataset(variant=variant)
        tok = LanoTokenizer()
        return ds, tok.total_vocab, ctx
    elif task == "capo":
        ctx = context_len_override or TASK_DEFAULT_CONTEXT["capo"]
        ds = build_capo_dataset(N=N)
        return ds, 256, ctx
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
        args.task, args.variant, args.N, args.K, args.L,
        context_len_override=args.context_len,
    )
    print(f"Context length: {context_len}")

    # num_workers=0 keeps data generation in the main process — much faster on MPS
    # where subprocess IPC overhead dominates over parallelism benefits at small batch sizes
    num_workers = 0 if device.type == "mps" else min(4, os.cpu_count())
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
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

    pbar = tqdm(total=args.max_steps, desc="Training", unit="step", dynamic_ncols=True)
    for batch in dataloader:
        if step >= args.max_steps:
            break

        # Update learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x = batch.to(device)
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
        pbar.update(1)

        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
            total_loss = 0.0

    pbar.close()

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
                   choices=["6L256D", "8L512D", "12L512D", "8L768D", "12L768D"],
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

    # Context length override (default: task-specific, e.g. 2048 for Depo)
    p.add_argument("--context_len", type=int, default=0,
                   help="Override context length (0 = use task default). "
                        "Smaller values are faster on MPS/CPU.")

    # Output
    p.add_argument("--save_path", default="", help="Path to save checkpoint")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
