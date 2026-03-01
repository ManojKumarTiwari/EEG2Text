#!/bin/bash
# ============================================================
# Step 1: Extract Kandinsky 2.2 image embedding targets
# ============================================================
# Submit:  sbatch eeg2image/sh/01_extract_targets.sh
# Local:   bash eeg2image/sh/01_extract_targets.sh
# ============================================================
#SBATCH --job-name=eeg2img_01_targets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=log/eeg2image/%x_%j.out
#SBATCH --error=log/eeg2image/%x_%j.err
# ── Edit these for your cluster ─────────────────────────────
##SBATCH --partition=gpu
##SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

# Navigate to project root (two levels up from eeg2image/sh/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"
echo "Working directory: $PROJECT_ROOT"

# ── Environment ──────────────────────────────────────────────
# Uncomment and edit one of the following:
# source activate YOUR_CONDA_ENV
# source .venv/bin/activate

mkdir -p log/eeg2image

echo "================================================"
echo "Step 1: Extract Kandinsky targets"
echo "Job: ${SLURM_JOB_ID:-local}  |  $(date)"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-auto-detected}"
echo "================================================"

python eeg2image/extract_targets.py \
    --cache_dir  "cache/eeg_image" \
    --prior_model "kandinsky-community/kandinsky-2-2-prior" \
    --prior_steps 25 \
    --guidance    4.0 \
    --n_seeds     3

echo "================================================"
echo "Step 1 done at $(date)"
echo "================================================"
