#!/bin/bash
#SBATCH --job-name=baseline_depo1_K8
#SBATCH --partition gpu_h200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=perryn.chang@yale.edu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Single 8L768D baseline run at N=375, K=8.
# Purpose: check whether evaluate_depo inflates accuracy to ~40% via the ANS
# token effect, or whether the baseline genuinely scores near 0% as in the paper.

module reset
module load miniconda

if [ ! -f "$HOME/.conda/envs/my_env/bin/pip" ]; then
    rm -rf "$HOME/.conda/envs/my_env"
    conda create -n my_env python=3.11 pip -y
    "$HOME/.conda/envs/my_env/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu124
    "$HOME/.conda/envs/my_env/bin/pip" install tqdm transformers datasets
fi

mkdir -p "$SLURM_SUBMIT_DIR/logs"
mkdir -p "$SLURM_SUBMIT_DIR/checkpoints"

cd "$SLURM_SUBMIT_DIR/canon_layers"

$HOME/.conda/envs/my_env/bin/torchrun \
    --nproc_per_node=4 \
    --master-port=29600 \
    train.py \
    --task depo \
    --variant depo1 \
    --N 375 \
    --K 8 \
    --model_type transformer \
    --model_size 8L768D \
    --rope rope \
    --rope_fraction 1.0 \
    --seed 42 \
    --lr 0.001 \
    --weight_decay 0.03 \
    --batch_size 128 \
    --max_steps 112500 \
    --warmup_steps 1000 \
    --save_path "$SLURM_SUBMIT_DIR/checkpoints/baseline_depo1_N375_K8_8L768D_lr0.001_seed42.pt"
