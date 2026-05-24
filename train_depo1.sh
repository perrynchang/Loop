#!/bin/bash
#SBATCH --job-name=baseline_depo1
#SBATCH --partition gpu_h200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=perryn.chang@yale.edu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --array=0-3

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

# 4 LRs, one run each (best reported across the four per paper)
LRS=(0.0003 0.0005 0.001 0.002)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

echo "Run $SLURM_ARRAY_TASK_ID: lr=$LR"

$HOME/.conda/envs/my_env/bin/torchrun \
    --nproc_per_node=4 \
    --master-port=$((29500 + SLURM_ARRAY_TASK_ID)) \
    train.py \
    --task depo \
    --variant depo1 \
    --N 225 \
    --K 4 \
    --model_size 8L512D \
    --rope rope \
    --rope_fraction 1.0 \
    --lr $LR \
    --weight_decay 0.03 \
    --batch_size 128 \
    --max_steps 87500 \
    --warmup_steps 1000 \
    --save_path "$SLURM_SUBMIT_DIR/checkpoints/baseline_depo1_N225_lr${LR}.pt"
