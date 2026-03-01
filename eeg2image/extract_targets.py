"""
Step 1 — Extract Kandinsky 2.2 image embedding targets.

For each motor-imagery class this script:
  1. Encodes CLASS_DESCRIPTIONS via the Kandinsky 2.2 prior → 1024-d image embeddings
  2. Averages across descriptions + seeds to get stable per-class targets
  3. Encodes VISUAL_PROMPTS via the same prior → Approach-1 prior embeds
  4. Saves both to CACHE_DIR

The prior is unloaded after this step; all subsequent scripts use the cached tensors.

Usage (from project root):
    python eeg2image/extract_targets.py [--args]
"""

import argparse
import gc
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg2image.config import (
    CACHE_DIR,
    CLASS_NAMES,
    IMAGE_EMBED_DIM,
    KANDINSKY_GUIDANCE,
    KANDINSKY_PRIOR_MODEL,
    NUM_CLASSES,
    PRIOR_STEPS,
    get_device,
    make_dirs,
    print_gpu_info,
)
from eeg2image.prompts import CLASS_DESCRIPTIONS, VISUAL_PROMPTS


def parse_args():
    p = argparse.ArgumentParser(description="Extract Kandinsky image embedding targets")
    p.add_argument("--cache_dir",     default=CACHE_DIR)
    p.add_argument("--prior_model",   default=KANDINSKY_PRIOR_MODEL)
    p.add_argument("--prior_steps",   type=int,   default=PRIOR_STEPS)
    p.add_argument("--guidance",      type=float, default=KANDINSKY_GUIDANCE)
    p.add_argument("--n_seeds",       type=int,   default=3,
                   help="Seeds per description (averaged for stability)")
    p.add_argument("--force",         action="store_true",
                   help="Re-extract even if cache already exists")
    return p.parse_args()


def extract_targets(args, device):
    from diffusers import KandinskyV22PriorPipeline  # local import to keep startup fast

    targets_path = os.path.join(args.cache_dir, "kandinsky_class_targets.pt")
    approach1_path = os.path.join(args.cache_dir, "approach1_prior_embeds.pt")

    if not args.force and os.path.exists(targets_path) and os.path.exists(approach1_path):
        class_targets = torch.load(targets_path, map_location="cpu")
        approach1_embeds = torch.load(approach1_path, map_location="cpu")
        print(f"Loaded targets from cache. Shape: {class_targets.shape}")
        return class_targets, approach1_embeds

    print(f"Loading Kandinsky prior '{args.prior_model}' ...")
    pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        args.prior_model, torch_dtype=torch.float16
    ).to(device)
    pipe_prior.set_progress_bar_config(disable=True)
    print("Prior loaded.")
    print_gpu_info(device)

    # ── Per-class image embedding targets ─────────────────────────────────
    class_targets = torch.zeros(NUM_CLASSES, IMAGE_EMBED_DIM)
    for class_id, descs in CLASS_DESCRIPTIONS.items():
        all_embs = []
        for desc in descs:
            for seed in range(args.n_seeds):
                with torch.no_grad():
                    out = pipe_prior(
                        prompt=desc,
                        guidance_scale=args.guidance,
                        num_inference_steps=args.prior_steps,
                        generator=torch.Generator(device).manual_seed(seed),
                    )
                emb = out.image_embeds.cpu().float()   # (1, 1024)
                all_embs.append(emb)
        class_targets[class_id] = torch.cat(all_embs).mean(0)
        n = len(all_embs)
        print(f"  Class {class_id} ({CLASS_NAMES[class_id]}): {n} embeddings averaged "
              f"→ shape {class_targets[class_id].shape}")

    # ── Approach-1: prior embeds from visual text prompts ─────────────────
    approach1_embeds = {}
    for class_id, prompt in VISUAL_PROMPTS.items():
        with torch.no_grad():
            out = pipe_prior(
                prompt=prompt,
                guidance_scale=args.guidance,
                num_inference_steps=args.prior_steps,
                generator=torch.Generator(device).manual_seed(42),
            )
        approach1_embeds[class_id] = {
            "image_embeds":          out.image_embeds.cpu().float(),
            "negative_image_embeds": out.negative_image_embeds.cpu().float(),
        }
        print(f"  Approach-1 class {class_id} ({CLASS_NAMES[class_id]}): done")

    torch.save(class_targets,    targets_path)
    torch.save(approach1_embeds, approach1_path)
    print(f"\nSaved targets → {targets_path}")
    print(f"Saved Approach-1 embeds → {approach1_path}")

    del pipe_prior
    gc.collect()
    torch.cuda.empty_cache()

    return class_targets, approach1_embeds


def main():
    args   = parse_args()
    device = get_device()
    make_dirs(args.cache_dir)

    print(f"Device : {device}")
    print_gpu_info(device)

    class_targets, approach1_embeds = extract_targets(args, device)

    # L2-normalise and show inter-class similarities
    targets_norm = F.normalize(class_targets.float(), dim=-1)
    sim = targets_norm @ targets_norm.T
    print("\nInter-class Kandinsky image cosine similarities:")
    for i in range(NUM_CLASSES):
        row = f"  {CLASS_NAMES[i]:<12}: "
        row += "  ".join(f"{sim[i, j]:.3f}" for j in range(NUM_CLASSES))
        print(row)

    print("\nStep 1 complete — targets cached.")


if __name__ == "__main__":
    main()
