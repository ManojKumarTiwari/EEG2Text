"""
Step 5 — Evaluate the EEG projection and visualize results.

Produces:
  - Kandinsky-space nearest-neighbour classification accuracy (per class)
  - t-SNE plot of projected EEG embeddings (coloured by class)
  - 2×4 comparison grid: Approach 1 (text) vs Approach 2 (EEG class-average)
  - Per-sample gallery (Approach 2)
  - EEG feature + generated image side-by-side panels

Usage (from project root):
    python eeg2image/evaluate.py [--args]
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg2image.config import (
    CACHE_DIR,
    CLASS_NAMES,
    DROPOUT,
    EEG_DIM,
    IMAGE_EMBED_DIM,
    N_SAMPLES,
    NUM_CLASSES,
    PROJ_HIDDEN,
    SAVE_DIR,
    get_device,
    make_dirs,
)
from eeg2image.model import load_projection


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate EEG-to-image pipeline")
    p.add_argument("--cache_dir",      default=CACHE_DIR)
    p.add_argument("--save_dir",       default=SAVE_DIR)
    p.add_argument("--projection_ckpt", default=None)
    p.add_argument("--eeg_dim",        type=int,   default=EEG_DIM)
    p.add_argument("--clip_dim",       type=int,   default=IMAGE_EMBED_DIM)
    p.add_argument("--hidden",         type=int,   default=PROJ_HIDDEN)
    p.add_argument("--dropout",        type=float, default=DROPOUT)
    p.add_argument("--n_tsne",         type=int,   default=400,
                   help="Max samples for t-SNE plot")
    p.add_argument("--n_samples",      type=int,   default=N_SAMPLES)
    p.add_argument("--skip_plots",     action="store_true",
                   help="Skip comparison/gallery plots (useful on headless servers)")
    return p.parse_args()


# ── Metrics ───────────────────────────────────────────────────────────────
def nn_accuracy(proj_embs, test_labels, class_targets):
    """Nearest-neighbour accuracy in Kandinsky image embedding space."""
    sims     = proj_embs @ class_targets.T   # (N, 4)
    pred_ids = sims.argmax(dim=-1).cpu()
    correct  = (pred_ids == test_labels).sum().item()
    return pred_ids, correct / len(test_labels)


# ── Plots ─────────────────────────────────────────────────────────────────
def plot_tsne(proj_embs_np, labels_np, save_path, n_classes):
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    tsne    = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embs_2d = tsne.fit_transform(proj_embs_np)

    colors = cm.tab10(np.linspace(0, 1, n_classes))
    fig, ax = plt.subplots(figsize=(8, 7))
    for c in range(n_classes):
        mask = (labels_np == c)
        ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1],
                   color=colors[c], label=CLASS_NAMES[c],
                   alpha=0.6, s=30, edgecolors="none")
    ax.legend(fontsize=11, markerscale=1.5)
    ax.set_title("t-SNE of EEG→Kandinsky Projected Embeddings (test set)", fontsize=13)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE plot saved → {save_path}")


def plot_comparison_grid(approach1_images, approach2_images, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(22, 12))
    fig.suptitle(
        "EEG → Image Generation  |  BCIC-IV-2a Motor Imagery  |  Kandinsky 2.2 (Apache 2.0)",
        fontsize=15, fontweight="bold", y=1.01,
    )
    row_labels = [
        "Approach 1\n(Visual text prompt → Kandinsky)",
        "Approach 2\n(EEG → projected embed → Kandinsky)",
    ]
    for col in range(NUM_CLASSES):
        axes[0, col].imshow(approach1_images[col])
        axes[0, col].axis("off")
        axes[0, col].set_title(CLASS_NAMES[col], fontsize=13,
                               fontweight="bold", pad=8)
        axes[1, col].imshow(approach2_images[col])
        axes[1, col].axis("off")

    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, fontsize=10, rotation=90, labelpad=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison grid saved → {save_path}")


def plot_sample_gallery(sample_images, sample_labels_list, save_path, cols=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n    = len(sample_images)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    fig.suptitle(
        "Per-sample EEG-conditioned Generation (Approach 2)\n"
        "Each image is conditioned on a unique EEG trial's projected embedding",
        fontsize=13, fontweight="bold",
    )
    for i, (img, lbl) in enumerate(zip(sample_images, sample_labels_list)):
        axes[i].imshow(img)
        axes[i].set_title(f"Sample {i}\nTrue: {CLASS_NAMES[lbl]}", fontsize=10)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Sample gallery saved → {save_path}")


def plot_eeg_sidebyside(test_feats, sample_images, sample_labels_list,
                        save_path, n_show=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("EEG Signal → Generated Brain Visualization",
                 fontsize=14, fontweight="bold")
    gs = GridSpec(n_show, 2, figure=fig,
                  width_ratios=[2, 1], wspace=0.05, hspace=0.4)

    for i in range(n_show):
        lbl     = sample_labels_list[i]
        eeg_raw = test_feats[i].numpy()

        ax_eeg = fig.add_subplot(gs[i, 0])
        ax_eeg.plot(eeg_raw, linewidth=0.8, color="steelblue", alpha=0.8)
        ax_eeg.set_title(f"Sample {i}  |  True class: {CLASS_NAMES[lbl]}",
                         fontsize=10, loc="left")
        ax_eeg.set_xlabel("CSBrain pooled feature dim")
        ax_eeg.set_ylabel("Activation")
        ax_eeg.grid(True, alpha=0.3)

        ax_img = fig.add_subplot(gs[i, 1])
        ax_img.imshow(sample_images[i])
        ax_img.axis("off")
        ax_img.set_title("Generated image", fontsize=9)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Side-by-side plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device()
    make_dirs(args.save_dir)

    # ── Load cached artefacts ─────────────────────────────────────────────
    test_feats, test_labels = torch.load(
        os.path.join(args.cache_dir, "test_eeg_feats.pt"), map_location="cpu"
    )
    class_targets = torch.load(
        os.path.join(args.cache_dir, "kandinsky_class_targets.pt"),
        map_location="cpu",
    )
    class_targets = F.normalize(class_targets.float(), dim=-1).to(device)

    ckpt = args.projection_ckpt or os.path.join(
        args.save_dir, "best_eeg_image_projection.pth"
    )
    projection = load_projection(
        ckpt, device,
        in_dim=args.eeg_dim, clip_dim=args.clip_dim,
        hidden=args.hidden, dropout=args.dropout,
    )

    # ── NN Accuracy ───────────────────────────────────────────────────────
    with torch.no_grad():
        proj_embs = F.normalize(
            projection(test_feats.to(device)), dim=-1
        )  # (N, 1024)

    pred_ids, acc = nn_accuracy(proj_embs, test_labels, class_targets)
    total = len(test_labels)
    print(f"\nKandinsky-space NN accuracy : {acc:.4f}  "
          f"({int(acc*total)}/{total})")
    print(f"Chance level (4 classes)    : 0.2500")
    print("\nPer-class accuracy:")
    for c in range(NUM_CLASSES):
        mask  = (test_labels == c)
        c_acc = (pred_ids[mask] == c).float().mean().item()
        c_n   = mask.sum().item()
        print(f"  {CLASS_NAMES[c]:<12}: {c_acc:.4f}  ({int(c_acc*c_n)}/{c_n})")

    # ── Alignment to targets ──────────────────────────────────────────────
    class_avg_embs = torch.zeros(NUM_CLASSES, args.clip_dim)
    for c in range(NUM_CLASSES):
        mask = (test_labels == c)
        class_avg_embs[c] = F.normalize(
            proj_embs.cpu()[mask].mean(0, keepdim=True), dim=-1
        ).squeeze()

    print("\nAlignment (projected EEG class avg vs Kandinsky targets):")
    align = class_avg_embs @ class_targets.cpu().T
    for i in range(NUM_CLASSES):
        best = CLASS_NAMES[align[i].argmax().item()]
        row  = f"  {CLASS_NAMES[i]:<12}: "
        row += "  ".join(f"{align[i,j]:.3f}" for j in range(NUM_CLASSES))
        row += f"  → {best}"
        print(row)

    # ── t-SNE ─────────────────────────────────────────────────────────────
    try:
        n_plot = min(args.n_tsne, len(test_feats))
        idx    = torch.randperm(len(test_feats))[:n_plot]
        with torch.no_grad():
            embs_np = projection(test_feats[idx].to(device)).cpu().float().numpy()
        lbls_np = test_labels[idx].numpy()

        plot_tsne(embs_np, lbls_np,
                  os.path.join(args.save_dir, "tsne_clip_embeddings.png"),
                  NUM_CLASSES)
    except ImportError:
        print("scikit-learn not installed — skipping t-SNE.")
    except Exception as e:
        print(f"t-SNE failed: {e}")

    # ── Comparison grid + gallery ─────────────────────────────────────────
    if not args.skip_plots:
        def _load_images(folder, suffixes):
            from PIL import Image
            imgs = {}
            for c in range(NUM_CLASSES):
                path = os.path.join(folder, suffixes[c])
                if os.path.exists(path):
                    imgs[c] = Image.open(path)
            return imgs

        a1_dir = os.path.join(args.save_dir, "approach1_text")
        a2_dir = os.path.join(args.save_dir, "approach2_eeg")
        smp_dir = os.path.join(args.save_dir, "approach2_samples")

        a1_files = [
            f"class_{c}_{CLASS_NAMES[c].replace(' ', '_')}.png"
            for c in range(NUM_CLASSES)
        ]
        a2_files = [
            f"class_{c}_{CLASS_NAMES[c].replace(' ', '_')}_eeg.png"
            for c in range(NUM_CLASSES)
        ]

        approach1_imgs = _load_images(a1_dir, a1_files)
        approach2_imgs = _load_images(a2_dir, a2_files)

        if len(approach1_imgs) == NUM_CLASSES and len(approach2_imgs) == NUM_CLASSES:
            plot_comparison_grid(
                approach1_imgs, approach2_imgs,
                os.path.join(args.save_dir, "comparison_grid.png"),
            )
        else:
            print("Skipping comparison grid (some images missing — run generate.py first).")

        # Per-sample gallery
        from PIL import Image
        n = min(args.n_samples, len(test_feats))
        sample_images, sample_labels_list = [], []
        for i in range(n):
            lbl = test_labels[i].item()
            path = os.path.join(
                smp_dir,
                f"sample_{i:03d}_class_{CLASS_NAMES[lbl].replace(' ', '_')}.png",
            )
            if os.path.exists(path):
                sample_images.append(Image.open(path))
                sample_labels_list.append(lbl)

        if sample_images:
            plot_sample_gallery(
                sample_images, sample_labels_list,
                os.path.join(args.save_dir, "sample_gallery.png"),
            )
            plot_eeg_sidebyside(
                test_feats, sample_images, sample_labels_list,
                os.path.join(args.save_dir, "eeg_to_image_sidebyside.png"),
                n_show=min(4, len(sample_images)),
            )
        else:
            print("Skipping gallery/side-by-side (no sample images found).")

    print("\nStep 5 complete — evaluation done.")


if __name__ == "__main__":
    main()
