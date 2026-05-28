#!/bin/bash
#SBATCH --job-name=loop_depo1
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
#SBATCH --array=0-1

# Array layout: 2 model configs, 1 LR, 1 seed = 2 runs
#   idx 0: 2L768D T_max=4  — T_max = K = 4, perfect loop-to-hop alignment
#   idx 1: 4L768D T_max=2  — T_max < K = 4, internal comparison
#
# K=4 so that T_max=4 aligns with max hop depth.
# isoFLOP: both configs = 8 layer-applications per forward pass = 8L768D baseline.

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

MODEL_SIZES=("2L768D" "4L768D")
T_MAXS=(4 2)

MODEL_SIZE=${MODEL_SIZES[$SLURM_ARRAY_TASK_ID]}
T_MAX=${T_MAXS[$SLURM_ARRAY_TASK_ID]}

echo "Run $SLURM_ARRAY_TASK_ID: model=$MODEL_SIZE T_max=$T_MAX K=4 lr=0.001 seed=42"

SAVE_PATH="$SLURM_SUBMIT_DIR/checkpoints/loop_depo1_N375_K4_${MODEL_SIZE}_T${T_MAX}_deepsup_lr0.001_seed42.pt"

$HOME/.conda/envs/my_env/bin/torchrun \
    --nproc_per_node=4 \
    --master-port=$((29500 + SLURM_ARRAY_TASK_ID)) \
    train.py \
    --task depo \
    --variant depo1 \
    --N 375 \
    --K 4 \
    --model_type loop \
    --model_size $MODEL_SIZE \
    --T_max $T_MAX \
    --loop_objective deep_sup \
    --rope rope \
    --rope_fraction 1.0 \
    --seed 42 \
    --lr 0.001 \
    --weight_decay 0.03 \
    --batch_size 128 \
    --max_steps 112500 \
    --warmup_steps 1000 \
    --save_path "$SAVE_PATH"
