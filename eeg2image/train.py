"""
Step 3 — Train the EEG→Kandinsky image embedding projection.

Trains a 3-layer MLP (EEGImageProjection) that maps the 200-d globally-pooled
CSBrain features into the 1024-d Kandinsky 2.2 image embedding space using:
  - MSE loss   (exact regression)
  - Cosine similarity loss (directional alignment)

Reads cached EEG features (Step 2) and Kandinsky targets (Step 1).
Only the projection MLP is updated; CSBrain stays frozen.

Usage (from project root):
    python eeg2image/train.py [--args]
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg2image.config import (
    BATCH_SIZE,
    CACHE_DIR,
    CLASS_NAMES,
    DROPOUT,
    EEG_DIM,
    EPOCHS,
    IMAGE_EMBED_DIM,
    LR,
    NUM_CLASSES,
    PROJ_HIDDEN,
    SAVE_DIR,
    WEIGHT_DECAY,
    get_device,
    make_dirs,
    print_gpu_info,
)
from eeg2image.model import EEGImageProjection, build_projection, save_projection


def parse_args():
    p = argparse.ArgumentParser(description="Train EEG→CLIP projection")
    p.add_argument("--cache_dir",    default=CACHE_DIR)
    p.add_argument("--save_dir",     default=SAVE_DIR)
    p.add_argument("--epochs",       type=int,   default=EPOCHS)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--eeg_dim",      type=int,   default=EEG_DIM)
    p.add_argument("--clip_dim",     type=int,   default=IMAGE_EMBED_DIM)
    p.add_argument("--hidden",       type=int,   default=PROJ_HIDDEN)
    p.add_argument("--dropout",      type=float, default=DROPOUT)
    p.add_argument("--resume",       default=None,
                   help="Path to checkpoint to resume training from")
    return p.parse_args()


def train_one_epoch(projection, loader, class_targets, optimizer, device):
    projection.train()
    total_loss = 0.0
    for feats, labels in loader:
        feats  = feats.to(device)
        labels = labels.to(device)

        targets = class_targets[labels]          # (B, 1024) L2-normalised targets
        preds   = projection(feats)              # (B, 1024)
        preds_n = F.normalize(preds, dim=-1)

        mse_loss = F.mse_loss(preds, targets)
        cos_loss = (1.0 - F.cosine_similarity(preds_n, targets, dim=-1)).mean()
        loss     = 0.5 * mse_loss + 0.5 * cos_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(projection, loader, class_targets, device):
    projection.eval()
    cos_scores = []
    for feats, labels in loader:
        preds = F.normalize(projection(feats.to(device)), dim=-1)
        tgts  = class_targets[labels.to(device)]
        cos_scores.append(F.cosine_similarity(preds, tgts, dim=-1).mean().item())
    return float(np.mean(cos_scores))


def main():
    args   = parse_args()
    device = get_device()
    make_dirs(args.save_dir)

    print(f"Device : {device}")
    print_gpu_info(device)

    # ── Load cached features ───────────────────────────────────────────────
    train_feats, train_labels = torch.load(
        os.path.join(args.cache_dir, "train_eeg_feats.pt")
    )
    val_feats, val_labels = torch.load(
        os.path.join(args.cache_dir, "val_eeg_feats.pt")
    )
    class_targets = torch.load(
        os.path.join(args.cache_dir, "kandinsky_class_targets.pt"),
        map_location="cpu",
    )
    # L2-normalise targets
    class_targets = F.normalize(class_targets.float(), dim=-1).to(device)

    print(f"Train: {train_feats.shape} | Val: {val_feats.shape}")
    print(f"Targets: {class_targets.shape}")

    train_loader = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=args.batch_size, shuffle=True, drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(val_feats, val_labels),
        batch_size=args.batch_size, shuffle=False,
    )

    # ── Build model ────────────────────────────────────────────────────────
    projection = build_projection(
        in_dim=args.eeg_dim, clip_dim=args.clip_dim,
        hidden=args.hidden, dropout=args.dropout,
    ).to(device)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        projection.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from '{args.resume}'.")

    optimizer = torch.optim.AdamW(
        projection.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=5e-6
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_cos = -1.0
    best_state   = None
    train_losses, val_coses = [], []
    ckpt_path = os.path.join(args.save_dir, "best_eeg_image_projection.pth")

    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Val cos':>8}  {'LR':>10}")
    print("-" * 44)

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(
            projection, train_loader, class_targets, optimizer, device
        )
        val_cos  = evaluate(projection, val_loader, class_targets, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        train_losses.append(avg_loss)
        val_coses.append(val_cos)

        marker = ""
        if val_cos > best_val_cos:
            best_val_cos = val_cos
            best_state   = copy.deepcopy(projection.state_dict())
            save_projection(projection, ckpt_path)
            marker = " *"

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:>6}  {avg_loss:>8.4f}  {val_cos:>8.4f}  "
                  f"{lr_now:>10.2e}{marker}")

    print(f"\nBest val cosine similarity: {best_val_cos:.4f}")
    print(f"Checkpoint saved → {ckpt_path}")

    # ── Save training curves ───────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(train_losses, color="steelblue")
        ax1.set_title("Training Loss (MSE + Cosine)")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        ax2.plot(val_coses, color="darkorange")
        ax2.set_title("Validation Cosine Similarity to Kandinsky Target")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Cosine Similarity")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        curve_path = os.path.join(args.save_dir, "training_curve.png")
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Training curve saved → {curve_path}")
    except Exception as e:
        print(f"Could not save training curve: {e}")

    # ── Reload best and print per-class alignment ──────────────────────────
    projection.load_state_dict(best_state)
    projection.eval()

    test_feats, test_labels = torch.load(
        os.path.join(args.cache_dir, "test_eeg_feats.pt")
    )
    with torch.no_grad():
        proj_embs = F.normalize(projection(test_feats.to(device)), dim=-1)

    sims     = proj_embs @ class_targets.T
    pred_ids = sims.argmax(dim=-1).cpu()
    acc      = (pred_ids == test_labels).float().mean().item()
    print(f"\nTest NN accuracy (Kandinsky space): {acc:.4f}  (chance 0.25)")

    print("\nPer-class:")
    for c in range(NUM_CLASSES):
        mask   = (test_labels == c)
        c_acc  = (pred_ids[mask] == c).float().mean().item()
        c_n    = mask.sum().item()
        print(f"  {CLASS_NAMES[c]:<12}: {c_acc:.4f}  ({int(c_acc*c_n)}/{c_n})")

    print("\nStep 3 complete — projection trained.")


if __name__ == "__main__":
    main()
