"""
Step 4 — Generate brain-visualization images using Kandinsky 2.2.

Approach 1 (text-guided):
    Visual text prompt → Kandinsky prior image embed (cached) → decoder → image

Approach 2 (EEG-conditioned):
    EEG → CSBrain → pool → EEGImageProjection → 1024-d embed → decoder → image

Usage (from project root):
    python eeg2image/generate.py [--args]
    python eeg2image/generate.py --approach 1       # text only
    python eeg2image/generate.py --approach 2       # EEG only
    python eeg2image/generate.py --approach both    # default
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg2image.config import (
    CACHE_DIR,
    CLASS_NAMES,
    DECODER_STEPS,
    EEG_DIM,
    IMAGE_EMBED_DIM,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    KANDINSKY_DECODER_MODEL,
    KANDINSKY_GUIDANCE,
    N_SAMPLES,
    NUM_CLASSES,
    PROJ_HIDDEN,
    SAVE_DIR,
    DROPOUT,
    get_device,
    make_dirs,
    print_gpu_info,
)
from eeg2image.model import load_projection


def parse_args():
    p = argparse.ArgumentParser(description="Generate EEG-conditioned images")
    p.add_argument("--cache_dir",      default=CACHE_DIR)
    p.add_argument("--save_dir",       default=SAVE_DIR)
    p.add_argument("--decoder_model",  default=KANDINSKY_DECODER_MODEL)
    p.add_argument("--decoder_steps",  type=int,   default=DECODER_STEPS)
    p.add_argument("--guidance",       type=float, default=KANDINSKY_GUIDANCE)
    p.add_argument("--image_height",   type=int,   default=IMAGE_HEIGHT)
    p.add_argument("--image_width",    type=int,   default=IMAGE_WIDTH)
    p.add_argument("--n_samples",      type=int,   default=N_SAMPLES,
                   help="Number of per-sample images to generate (Approach 2)")
    p.add_argument("--approach",       default="both",
                   choices=["1", "2", "both"],
                   help="Which generation approach to run")
    p.add_argument("--projection_ckpt", default=None,
                   help="Path to projection checkpoint (default: save_dir/best_eeg_image_projection.pth)")
    p.add_argument("--eeg_dim",        type=int, default=EEG_DIM)
    p.add_argument("--clip_dim",       type=int, default=IMAGE_EMBED_DIM)
    p.add_argument("--hidden",         type=int, default=PROJ_HIDDEN)
    p.add_argument("--dropout",        type=float, default=DROPOUT)
    return p.parse_args()


def _gen_image(pipe_decoder, img_emb, neg_emb, args, seed, device):
    with torch.no_grad():
        return pipe_decoder(
            image_embeds=img_emb.to(dtype=torch.float16, device=device),
            negative_image_embeds=neg_emb.to(dtype=torch.float16, device=device),
            height=args.image_height,
            width=args.image_width,
            num_inference_steps=args.decoder_steps,
            guidance_scale=args.guidance,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]


def run_approach1(pipe_decoder, args, device):
    """Generate one image per class from pre-cached prior embeds."""
    approach1_path = os.path.join(args.cache_dir, "approach1_prior_embeds.pt")
    if not os.path.exists(approach1_path):
        raise FileNotFoundError(
            f"Approach-1 prior embeds not found: '{approach1_path}'. "
            "Run extract_targets.py first."
        )

    approach1_embeds = torch.load(approach1_path, map_location="cpu")
    out_dir = os.path.join(args.save_dir, "approach1_text")
    make_dirs(out_dir)
    images = {}

    for class_id in range(NUM_CLASSES):
        embs    = approach1_embeds[class_id]
        img_emb = embs["image_embeds"]
        neg_emb = embs["negative_image_embeds"]

        print(f"  [Approach 1] Class {class_id} ({CLASS_NAMES[class_id]}) ...")
        image = _gen_image(pipe_decoder, img_emb, neg_emb, args, seed=42, device=device)

        fname = f"class_{class_id}_{CLASS_NAMES[class_id].replace(' ', '_')}.png"
        image.save(os.path.join(out_dir, fname))
        images[class_id] = image
        print(f"    Saved → {os.path.join(out_dir, fname)}")

    print("Approach 1 done.")
    return images


def run_approach2(pipe_decoder, projection, args, device):
    """Generate images from EEG-projected embeddings (class-average + per-sample)."""
    test_feats, test_labels = torch.load(
        os.path.join(args.cache_dir, "test_eeg_feats.pt"), map_location="cpu"
    )

    projection.eval()
    with torch.no_grad():
        all_embs = projection(test_feats.to(device)).cpu().float()  # (N, 1024)

    zero_neg = torch.zeros(1, args.clip_dim)

    # ── Class-average images ───────────────────────────────────────────────
    avg_dir = os.path.join(args.save_dir, "approach2_eeg")
    make_dirs(avg_dir)
    avg_images = {}

    for class_id in range(NUM_CLASSES):
        mask     = (test_labels == class_id)
        cls_emb  = all_embs[mask].mean(0, keepdim=True)   # (1, 1024)

        print(f"  [Approach 2 avg] Class {class_id} ({CLASS_NAMES[class_id]}) "
              f"n={mask.sum().item()} ...")
        image = _gen_image(pipe_decoder, cls_emb, zero_neg, args,
                           seed=42 + class_id, device=device)

        fname = f"class_{class_id}_{CLASS_NAMES[class_id].replace(' ', '_')}_eeg.png"
        image.save(os.path.join(avg_dir, fname))
        avg_images[class_id] = image
        print(f"    Saved → {os.path.join(avg_dir, fname)}")

    print("Approach 2 class-average images done.")

    # ── Per-sample images ──────────────────────────────────────────────────
    samples_dir = os.path.join(args.save_dir, "approach2_samples")
    make_dirs(samples_dir)
    sample_images, sample_labels_list = [], []
    n = min(args.n_samples, len(test_feats))

    for i in range(n):
        emb       = all_embs[i].unsqueeze(0)   # (1, 1024)
        true_lbl  = test_labels[i].item()

        image = _gen_image(pipe_decoder, emb, zero_neg, args,
                           seed=100 + i, device=device)

        sample_images.append(image)
        sample_labels_list.append(true_lbl)
        fname = f"sample_{i:03d}_class_{CLASS_NAMES[true_lbl].replace(' ', '_')}.png"
        image.save(os.path.join(samples_dir, fname))
        print(f"  Sample {i:2d} | true: {CLASS_NAMES[true_lbl]:<12} → saved")

    print("Approach 2 per-sample images done.")
    return avg_images, sample_images, sample_labels_list


def main():
    args   = parse_args()
    device = get_device()
    make_dirs(args.save_dir)

    print(f"Device : {device}")
    print_gpu_info(device)

    # ── Load Kandinsky 2.2 decoder ─────────────────────────────────────────
    from diffusers import KandinskyV22Pipeline  # local import

    print(f"Loading Kandinsky 2.2 decoder '{args.decoder_model}' ...")
    pipe_decoder = KandinskyV22Pipeline.from_pretrained(
        args.decoder_model, torch_dtype=torch.float16
    ).to(device)
    pipe_decoder.set_progress_bar_config(disable=True)
    print("Decoder loaded.")
    print_gpu_info(device)

    # ── Load projection (needed for Approach 2) ────────────────────────────
    projection = None
    if args.approach in ("2", "both"):
        ckpt = args.projection_ckpt or os.path.join(
            args.save_dir, "best_eeg_image_projection.pth"
        )
        projection = load_projection(
            ckpt, device,
            in_dim=args.eeg_dim, clip_dim=args.clip_dim,
            hidden=args.hidden, dropout=args.dropout,
        )

    # ── Generate ───────────────────────────────────────────────────────────
    approach1_images = None
    approach2_avg    = None

    if args.approach in ("1", "both"):
        approach1_images = run_approach1(pipe_decoder, args, device)

    if args.approach in ("2", "both"):
        approach2_avg, sample_images, sample_labels_list = run_approach2(
            pipe_decoder, projection, args, device
        )

    print(f"\nStep 4 complete — images saved to '{args.save_dir}'.")


if __name__ == "__main__":
    main()
