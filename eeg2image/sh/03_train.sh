#!/bin/bash
# ============================================================
# Step 3: Train EEG → Kandinsky image embedding projection
# ============================================================
# Submit:  sbatch eeg2image/sh/03_train.sh
# Local:   bash eeg2image/sh/03_train.sh
# ============================================================
#SBATCH --job-name=eeg2img_03_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=log/eeg2image/%x_%j.out
#SBATCH --error=log/eeg2image/%x_%j.err
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"
echo "Working directory: $PROJECT_ROOT"

# source activate YOUR_CONDA_ENV
mkdir -p log/eeg2image

echo "================================================"
echo "Step 3: Train EEG projection"
echo "Job: ${SLURM_JOB_ID:-local}  |  $(date)"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-auto-detected}"
echo "================================================"

python eeg2image/train.py \
    --cache_dir    "cache/eeg_image" \
    --save_dir     "pth_downtasks/eeg_image" \
    --epochs       40 \
    --lr           5e-4 \
    --weight_decay 0.01 \
    --batch_size   256 \
    --eeg_dim      200 \
    --clip_dim     1024 \
    --hidden       1024 \
    --dropout      0.1

echo "================================================"
echo "Step 3 done at $(date)"
echo "================================================"
