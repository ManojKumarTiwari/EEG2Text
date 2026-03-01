#!/bin/bash
# ============================================================
# Step 5: Evaluate projection quality & produce all plots
# ============================================================
# Submit:  sbatch eeg2image/sh/05_evaluate.sh
# Local:   bash eeg2image/sh/05_evaluate.sh
# ============================================================
#SBATCH --job-name=eeg2img_05_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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
echo "Step 5: Evaluate"
echo "Job: ${SLURM_JOB_ID:-local}  |  $(date)"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-auto-detected}"
echo "================================================"

python eeg2image/evaluate.py \
    --cache_dir  "cache/eeg_image" \
    --save_dir   "pth_downtasks/eeg_image" \
    --eeg_dim    200 \
    --clip_dim   1024 \
    --hidden     1024 \
    --dropout    0.1 \
    --n_tsne     400 \
    --n_samples  8

echo "================================================"
echo "Step 5 done at $(date)"
echo "================================================"
