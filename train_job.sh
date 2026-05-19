#!/bin/bash
#SBATCH --job-name=baseline_depo1
#SBATCH --partition gpu_h200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=perryn.chang@yale.edu
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module reset
module load miniconda

if [ ! -f "$HOME/.conda/envs/my_env/bin/pip" ]; then
    rm -rf "$HOME/.conda/envs/my_env"
    conda create -n my_env python=3.11 pip -y
fi

ENVPIP="$HOME/.conda/envs/my_env/bin/pip"
ENVPY="$HOME/.conda/envs/my_env/bin/python"

$ENVPIP install torch --index-url https://download.pytorch.org/whl/cu124
$ENVPIP install tqdm transformers datasets

cd "$SLURM_SUBMIT_DIR/canon_layers"

$HOME/.conda/envs/my_env/bin/torchrun --nproc_per_node=4 train.py \
    --task depo \
    --variant depo1 \
    --N 225 \
    --K 4 \
    --model_size 8L512D \
    --rope rope \
    --rope_fraction 1.0 \
    --lr 0.0003 \
    --weight_decay 0.03 \
    --batch_size 128 \
    --max_steps 87500 \
    --warmup_steps 1000 \
    --context_len 2048 \
    --save_path checkpoints/baseline_depo1.pt