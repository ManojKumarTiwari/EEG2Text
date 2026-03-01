"""EEGImageProjection: maps pooled CSBrain features → Kandinsky image embedding space."""

import os
import torch
import torch.nn as nn


class EEGImageProjection(nn.Module):
    """3-layer MLP: EEG (200-d) → Kandinsky 2.2 image embedding (1024-d).

    Training objective
    ------------------
    MSE + cosine-similarity loss against Kandinsky prior image embeddings
    extracted from class motor-imagery descriptions.  No paired EEG-image
    data is required.

    Inference
    ---------
    Output is passed to the Kandinsky 2.2 decoder as `image_embeds`.
    """

    def __init__(
        self,
        in_dim:   int = 200,
        clip_dim: int = 1024,
        hidden:   int = 1024,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, clip_dim),
            nn.LayerNorm(clip_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_dim) → (batch, clip_dim)"""
        return self.net(x)


def build_projection(
    in_dim:   int = 200,
    clip_dim: int = 1024,
    hidden:   int = 1024,
    dropout:  float = 0.1,
) -> EEGImageProjection:
    model = EEGImageProjection(in_dim, clip_dim, hidden, dropout)
    n = sum(p.numel() for p in model.parameters())
    print(f"EEGImageProjection — {n:,} parameters")
    return model


def load_projection(checkpoint_path: str, device: torch.device,
                    **kwargs) -> EEGImageProjection:
    """Load a trained projection from a checkpoint file."""
    model = build_projection(**kwargs)
    sd = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    print(f"Loaded projection from '{checkpoint_path}'.")
    return model


def save_projection(model: EEGImageProjection, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
