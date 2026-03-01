"""
Step 2 — Extract and cache EEG features via the frozen CSBrain encoder.

Runs all three data splits (train / val / test) through CSBrain once and
saves the globally-pooled 200-d features to CACHE_DIR.  Subsequent scripts
load these tensors directly, skipping the 12-layer transformer at every epoch.

Usage (from project root):
    python eeg2image/extract_features.py [--args]
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg2image.config import (
    BATCH_SIZE,
    CACHE_DIR,
    CSBRAIN_WEIGHTS_PATH,
    LMDB_DIR,
    get_device,
    make_dirs,
    print_gpu_info,
)
from eeg2image.dataset import get_dataloaders
from eeg2image.encoder import build_encoder, pool_eeg


def parse_args():
    p = argparse.ArgumentParser(description="Extract CSBrain EEG features")
    p.add_argument("--lmdb_dir",   default=LMDB_DIR)
    p.add_argument("--cache_dir",  default=CACHE_DIR)
    p.add_argument("--weights",    default=CSBRAIN_WEIGHTS_PATH)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--force",      action="store_true",
                   help="Re-extract even if cache already exists")
    return p.parse_args()


def extract_split(loader, encoder, reducer, device, desc="Extracting"):
    all_feats, all_labels = [], []
    encoder.eval()
    reducer.eval()
    for eeg, labels in tqdm(loader, desc=desc, mininterval=10):
        eeg = eeg.to(device)
        pooled = pool_eeg(encoder, reducer, eeg)  # (B, 200)
        all_feats.append(pooled.cpu())
        all_labels.append(labels)
    return torch.cat(all_feats), torch.cat(all_labels)


def main():
    args   = parse_args()
    device = get_device()
    make_dirs(args.cache_dir)

    print(f"Device : {device}")
    print_gpu_info(device)

    train_path = os.path.join(args.cache_dir, "train_eeg_feats.pt")
    val_path   = os.path.join(args.cache_dir, "val_eeg_feats.pt")
    test_path  = os.path.join(args.cache_dir, "test_eeg_feats.pt")

    if not args.force and all(
        os.path.exists(p) for p in [train_path, val_path, test_path]
    ):
        train_feats, train_labels = torch.load(train_path)
        val_feats,   val_labels   = torch.load(val_path)
        test_feats,  test_labels  = torch.load(test_path)
        print("Loaded EEG features from cache.")
    else:
        train_loader, val_loader, test_loader, *_ = get_dataloaders(
            args.lmdb_dir, batch_size=args.batch_size, num_workers=args.num_workers
        )
        print(f"Train {len(train_loader.dataset)} | "
              f"Val {len(val_loader.dataset)} | "
              f"Test {len(test_loader.dataset)}")

        encoder, reducer = build_encoder(args.weights)
        encoder = encoder.to(device)
        reducer = reducer.to(device)

        train_feats, train_labels = extract_split(
            train_loader, encoder, reducer, device, "Train"
        )
        val_feats, val_labels = extract_split(
            val_loader, encoder, reducer, device, "Val"
        )
        test_feats, test_labels = extract_split(
            test_loader, encoder, reducer, device, "Test"
        )

        torch.save((train_feats, train_labels), train_path)
        torch.save((val_feats,   val_labels),   val_path)
        torch.save((test_feats,  test_labels),  test_path)
        print("Extracted and saved EEG features.")

    print(f"Shapes — Train: {train_feats.shape} | "
          f"Val: {val_feats.shape} | Test: {test_feats.shape}")
    print("Step 2 complete — EEG features cached.")


if __name__ == "__main__":
    main()
