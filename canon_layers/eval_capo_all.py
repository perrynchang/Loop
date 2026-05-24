"""
Batch evaluation of all Capo checkpoints in ../checkpoints/capo_*.pt.
Loads the GPT-2 tokenizer and compact vocab once, then loops over every checkpoint.
"""
import glob
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

from evaluate import load_model, evaluate_capo
from tasks.capo import _get_compact_vocab  # warm the cache once

print("Building compact vocab (one-time) ...")
_get_compact_vocab()
print("Done.\n")

device_str = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_str)
print(f"Device: {device}\n")

ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, "capo_*.pt")))
print(f"Found {len(checkpoints)} Capo checkpoints.\n")

ATTR_KEYS = ['birthday', 'birthcity', 'university', 'field', 'company1name', 'company1city']

# Header
header = f"{'Checkpoint':<35} {'N':>7} {'Model':<8} {'LR':>7} {'NLL':>7} " + \
         "  ".join(f"{k[:6]:>6}" for k in ATTR_KEYS) + \
         f"  {'mean%':>6}  {'BPP':>8}  {'Params':>10}"
print(header)
print("-" * len(header))

rows = []
for ckpt_path in checkpoints:
    name = os.path.basename(ckpt_path).replace(".pt", "")  # e.g. capo_N50000_2L512D_lr0.001

    try:
        model, train_args = load_model(ckpt_path, device)
    except Exception as e:
        print(f"  ERROR loading {name}: {e}")
        continue

    try:
        results = evaluate_capo(model, train_args, N_eval=500, device=device)
    except Exception as e:
        print(f"  ERROR evaluating {name}: {e}")
        continue

    N          = train_args.get('N', '?')
    model_size = train_args.get('model_size', '?')
    lr         = train_args.get('lr', '?')
    nll        = results['mean_loss']
    mean_acc   = results['mean_accuracy'] * 100
    bpp        = results['bpp_estimate']
    n_params   = results['n_params']
    per_attr   = results['per_attr_accuracy']

    attr_str = "  ".join(f"{per_attr.get(k, 0)*100:>6.1f}" for k in ATTR_KEYS)
    row = (f"{name:<35} {N:>7} {model_size:<8} {lr:>7}  {nll:>6.4f}  {attr_str}  {mean_acc:>5.1f}%  {bpp:>8.4f}  {n_params:>10,}")
    print(row)
    rows.append((name, N, model_size, lr, nll, per_attr, mean_acc, bpp, n_params))

print("\nDone.")
