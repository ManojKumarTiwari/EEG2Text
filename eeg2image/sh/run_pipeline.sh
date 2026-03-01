#!/bin/bash
# ============================================================
# EEG-to-Image full pipeline — SLURM job-dependency chain
#
# Submits all 5 steps as a dependent chain so each step
# only starts after the previous one succeeds.
#
# Usage (from project root):
#   bash eeg2image/sh/run_pipeline.sh
#
# To run locally (no SLURM) use:
#   bash eeg2image/sh/run_pipeline.sh --local
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

LOCAL=0
if [[ "${1:-}" == "--local" ]]; then
    LOCAL=1
fi

mkdir -p log/eeg2image

# ── Helper: extract job ID from sbatch output ──────────────
jobid_of() {
    # "Submitted batch job 12345" → 12345
    echo "$1" | awk '{print $NF}'
}

if [[ $LOCAL -eq 1 ]]; then
    echo "============================="
    echo " Running pipeline locally"
    echo "============================="

    echo "[1/5] Extracting Kandinsky targets..."
    bash "$SCRIPT_DIR/01_extract_targets.sh"

    echo "[2/5] Extracting EEG features..."
    bash "$SCRIPT_DIR/02_extract_features.sh"

    echo "[3/5] Training projection..."
    bash "$SCRIPT_DIR/03_train.sh"

    echo "[4/5] Generating images..."
    bash "$SCRIPT_DIR/04_generate.sh"

    echo "[5/5] Evaluating..."
    bash "$SCRIPT_DIR/05_evaluate.sh"

    echo "============================="
    echo " Pipeline complete!"
    echo " Outputs → pth_downtasks/eeg_image/"
    echo "============================="

else
    echo "========================================"
    echo " Submitting EEG-to-Image SLURM pipeline"
    echo "========================================"

    # Step 1 — no dependency
    OUT1=$(sbatch "$SCRIPT_DIR/01_extract_targets.sh")
    JID1=$(jobid_of "$OUT1")
    echo "Step 1 submitted: job $JID1  ($OUT1)"

    # Step 2 — after step 1 succeeds
    OUT2=$(sbatch --dependency=afterok:"$JID1" "$SCRIPT_DIR/02_extract_features.sh")
    JID2=$(jobid_of "$OUT2")
    echo "Step 2 submitted: job $JID2  (depends on $JID1)"

    # Step 3 — after step 2 succeeds
    OUT3=$(sbatch --dependency=afterok:"$JID2" "$SCRIPT_DIR/03_train.sh")
    JID3=$(jobid_of "$OUT3")
    echo "Step 3 submitted: job $JID3  (depends on $JID2)"

    # Step 4 — after step 3 succeeds
    OUT4=$(sbatch --dependency=afterok:"$JID3" "$SCRIPT_DIR/04_generate.sh")
    JID4=$(jobid_of "$OUT4")
    echo "Step 4 submitted: job $JID4  (depends on $JID3)"

    # Step 5 — after step 4 succeeds
    OUT5=$(sbatch --dependency=afterok:"$JID4" "$SCRIPT_DIR/05_evaluate.sh")
    JID5=$(jobid_of "$OUT5")
    echo "Step 5 submitted: job $JID5  (depends on $JID4)"

    echo ""
    echo "========================================"
    echo " All jobs submitted."
    echo " Monitor: squeue -u $USER"
    echo " Logs   : log/eeg2image/"
    echo " Outputs: pth_downtasks/eeg_image/"
    echo "========================================"
    echo ""
    echo " Job chain: $JID1 → $JID2 → $JID3 → $JID4 → $JID5"
fi
