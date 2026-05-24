#!/bin/bash
#SBATCH --job-name=baseline_mano
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
#SBATCH --array=0-7

module reset
module load miniconda

if [ ! -f "$HOME/.conda/envs/my_env/bin/pip" ]; then
    rm -rf "$HOME/.conda/envs/my_env"
    conda create -n my_env python=3.11 pip -y
    "$HOME/.conda/envs/my_env/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu124
    "$HOME/.conda/envs/my_env/bin/pip" install tqdm transformers datasets
fi

mkdir -p "$SLURM_SUBMIT_DIR/logs"

cd "$SLURM_SUBMIT_DIR/canon_layers"

# 4 LRs x 2 seeds = 8 runs (SLURM_ARRAY_TASK_ID 0..7)
LRS=(0.0001 0.0001 0.0002 0.0002 0.0003 0.0003 0.0005 0.0005)
SEEDS=(42 123 42 123 42 123 42 123)

LR=${LRS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Run $SLURM_ARRAY_TASK_ID: lr=$LR seed=$SEED"

$HOME/.conda/envs/my_env/bin/torchrun --nproc_per_node=4 --master-port=$((29500 + SLURM_ARRAY_TASK_ID)) train.py \
    --task mano \
    --L 10 \
    --model_size 8L512D \
    --rope rope \
    --rope_fraction 1.0 \
    --lr $LR \
    --seed $SEED \
    --weight_decay 0.1 \
    --batch_size 64 \
    --max_steps 80000 \
    --warmup_steps 1000 \
    --save_path checkpoints/baseline_mano_L10_lr${LR}_seed${SEED}.pt
