"""
Shared configuration defaults for the EEG-to-Image pipeline.

All paths are relative to the CSBrain project root.
Run scripts from the project root:
    python eeg2image/train.py [--args]
"""

import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────
CSBRAIN_WEIGHTS_PATH = "pth/CSBrain.pth"
LMDB_DIR             = "data/BCICIV2a/processed_lmdb"
SAVE_DIR             = "pth_downtasks/eeg_image"
CACHE_DIR            = "cache/eeg_image"

# ── Kandinsky 2.2 (Apache 2.0) ────────────────────────────────────────────
KANDINSKY_PRIOR_MODEL   = "kandinsky-community/kandinsky-2-2-prior"
KANDINSKY_DECODER_MODEL = "kandinsky-community/kandinsky-2-2-decoder"

# ── Architecture ──────────────────────────────────────────────────────────
NUM_CLASSES     = 4
EEG_DIM         = 200   # CSBrain pooled output dim
IMAGE_EMBED_DIM = 1024  # Kandinsky 2.2 prior outputs 1024-d (NOT 1280)
PROJ_HIDDEN     = 1024
DROPOUT         = 0.1

# ── Training ──────────────────────────────────────────────────────────────
BATCH_SIZE   = 64
EPOCHS       = 40
LR           = 5e-4
WEIGHT_DECAY = 0.01

# ── Image generation ──────────────────────────────────────────────────────
IMAGE_HEIGHT       = 512
IMAGE_WIDTH        = 512
PRIOR_STEPS        = 25
DECODER_STEPS      = 50
KANDINSKY_GUIDANCE = 4.0
N_SAMPLES          = 8   # number of per-sample test images to generate

# ── Class labels ──────────────────────────────────────────────────────────
CLASS_NAMES = {0: "Left Hand", 1: "Right Hand", 2: "Both Feet", 3: "Tongue"}


# ── Helpers ───────────────────────────────────────────────────────────────
def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_gpu_info(device):
    if device.type == "cuda":
        free_gb  = torch.cuda.mem_get_info()[0] / 1e9
        total_gb = torch.cuda.mem_get_info()[1] / 1e9
        print(f"GPU  : {torch.cuda.get_device_name(device)}")
        print(f"VRAM : {free_gb:.1f} / {total_gb:.1f} GB free")


def make_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
