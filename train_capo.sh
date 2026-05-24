#!/bin/bash
#SBATCH --job-name=capo_scaling
#SBATCH --partition gpu_h200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=perryn.chang@yale.edu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --array=0-23

# 4 model sizes x 3 N values x 2 LRs = 24 runs
#
# Model sizes (paper's ℓ-h notation → our naming):
#   3-6 → 3L384D   5-6 → 5L384D   6-6 → 6L384D   2-8 → 2L512D
#
# Layout (SLURM_ARRAY_TASK_ID):
#  0- 7 : N=50K
#  8-15 : N=100K
# 16-23 : N=200K
#
# Within each N block (+offset):
#  +0: 2L512D  lr=0.001    +1: 2L512D  lr=0.0005
#  +2: 3L384D  lr=0.001    +3: 3L384D  lr=0.0005
#  +4: 5L384D  lr=0.001    +5: 5L384D  lr=0.0005
#  +6: 6L384D  lr=0.001    +7: 6L384D  lr=0.0005
#
# Batch size scales with N to keep step count ~416K for all runs:
#  N=50K  -> batch_size=12  (3 per GPU)
#  N=100K -> batch_size=24  (6 per GPU)
#  N=200K -> batch_size=48  (12 per GPU)

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

NS=(          50000  50000  50000  50000  50000  50000  50000  50000  \
             100000 100000 100000 100000 100000 100000 100000 100000  \
             200000 200000 200000 200000 200000 200000 200000 200000)
MODEL_SIZES=(2L512D 2L512D 3L384D 3L384D 5L384D 5L384D 6L384D 6L384D \
             2L512D 2L512D 3L384D 3L384D 5L384D 5L384D 6L384D 6L384D \
             2L512D 2L512D 3L384D 3L384D 5L384D 5L384D 6L384D 6L384D)
LRS=(        0.001  0.0005 0.001  0.0005 0.001  0.0005 0.001  0.0005 \
             0.001  0.0005 0.001  0.0005 0.001  0.0005 0.001  0.0005 \
             0.001  0.0005 0.001  0.0005 0.001  0.0005 0.001  0.0005)
BATCH_SIZES=(12     12     12     12     12     12     12     12     \
             24     24     24     24     24     24     24     24     \
             48     48     48     48     48     48     48     48)

N=${NS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODEL_SIZES[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
BATCH=${BATCH_SIZES[$SLURM_ARRAY_TASK_ID]}

echo "Run $SLURM_ARRAY_TASK_ID: N=$N model=$MODEL lr=$LR batch_size=$BATCH"

$HOME/.conda/envs/my_env/bin/torchrun \
    --nproc_per_node=4 \
    --master-port=$((29500 + SLURM_ARRAY_TASK_ID)) \
    train.py \
    --task capo \
    --N $N \
    --model_size $MODEL \
    --rope rope \
    --rope_fraction 1.0 \
    --tie_weights \
    --lr $LR \
    --weight_decay 0.01 \
    --batch_size $BATCH \
    --max_steps 500000 \
    --warmup_steps 1000 \
    --context_len 512 \
    --save_path $SLURM_SUBMIT_DIR/checkpoints/capo_N${N}_${MODEL}_lr${LR}.pt
