#!/bin/bash
# ============================================================
# Step 2: Extract & cache EEG features via frozen CSBrain
# ============================================================
# Submit:  sbatch eeg2image/sh/02_extract_features.sh
# Local:   bash eeg2image/sh/02_extract_features.sh
# ============================================================
#SBATCH --job-name=eeg2img_02_features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
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
echo "Step 2: Extract EEG features"
echo "Job: ${SLURM_JOB_ID:-local}  |  $(date)"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-auto-detected}"
echo "================================================"

python eeg2image/extract_features.py \
    --lmdb_dir   "data/BCICIV2a/processed_lmdb" \
    --cache_dir  "cache/eeg_image" \
    --weights    "pth/CSBrain.pth" \
    --batch_size 128 \
    --num_workers 4

echo "================================================"
echo "Step 2 done at $(date)"
echo "================================================"
