#!/bin/bash
# ============================================================
# Step 4: Generate brain-visualization images
#         Approach 1 (text)  +  Approach 2 (EEG-conditioned)
# ============================================================
# Submit:  sbatch eeg2image/sh/04_generate.sh
# Local:   bash eeg2image/sh/04_generate.sh
# ============================================================
#SBATCH --job-name=eeg2img_04_generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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
echo "Step 4: Generate images"
echo "Job: ${SLURM_JOB_ID:-local}  |  $(date)"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-auto-detected}"
echo "================================================"

python eeg2image/generate.py \
    --cache_dir      "cache/eeg_image" \
    --save_dir       "pth_downtasks/eeg_image" \
    --decoder_model  "kandinsky-community/kandinsky-2-2-decoder" \
    --decoder_steps  50 \
    --guidance       4.0 \
    --image_height   512 \
    --image_width    512 \
    --n_samples      8 \
    --approach       both \
    --eeg_dim        200 \
    --clip_dim       1024 \
    --hidden         1024 \
    --dropout        0.1

echo "================================================"
echo "Step 4 done at $(date)"
echo "================================================"
