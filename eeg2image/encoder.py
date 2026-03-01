"""CSBrain encoder loader and EEGTokenReducer for BCIC-IV-2a (22-channel layout)."""

import os
import sys
import torch
import torch.nn as nn

# Allow importing from project root (models/CSBrain.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.CSBrain import CSBrain  # noqa: E402

# ── BCIC-IV-2a 22-channel electrode layout ────────────────────────────────
_BCI42A_BRAIN_REGIONS = [
    0,
    4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4,
    1, 1, 1, 1,
]

_BCI42A_ELECTRODE_LABELS = [
    "Fz",
    "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C5",  "C3",  "C1",  "CZ",  "C2",  "C4",  "C6",
    "CP3", "CP1", "CPZ", "CP2", "CP4",
    "P1",  "PZ",  "P2",  "POZ",
]

_BCI42A_TOPOLOGY = {
    0: ["Fz"],
    4: ["FC3", "FC1", "FCZ", "FC2", "FC4", "C5", "C3", "C1",
        "CZ",  "C2",  "C4",  "C6",  "CP3", "CP1", "CPZ", "CP2", "CP4"],
    1: ["P1", "PZ", "P2", "POZ"],
}


def _build_sorted_indices():
    region_groups: dict = {}
    for i, region in enumerate(_BCI42A_BRAIN_REGIONS):
        region_groups.setdefault(region, []).append((i, _BCI42A_ELECTRODE_LABELS[i]))
    sorted_indices = []
    for region in sorted(region_groups.keys()):
        elecs = sorted(
            region_groups[region],
            key=lambda x: _BCI42A_TOPOLOGY[region].index(x[1]),
        )
        sorted_indices.extend([e[0] for e in elecs])
    return sorted_indices


class EEGTokenReducer(nn.Module):
    """Average EEG channels within each brain region.

    Input:  (batch, n_ch, n_patches, d_model)
    Output: (batch, n_regions, n_patches, d_model)
    """

    def __init__(self, area_config: dict):
        super().__init__()
        self.area_config = area_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = [
            x[:, self.area_config[r]["slice"], :, :].mean(1)
            for r in sorted(self.area_config)
        ]
        return torch.stack(tokens, dim=1)


def build_encoder(weights_path: str):
    """Build and return a frozen CSBrain encoder for the 22-ch BCIC-IV-2a layout.

    Returns
    -------
    encoder     : CSBrain (frozen, on CPU)
    reducer     : EEGTokenReducer (on CPU)
    """
    sorted_indices = _build_sorted_indices()

    encoder = CSBrain(
        in_dim=200, out_dim=200, d_model=200,
        dim_feedforward=800, seq_len=30,
        n_layer=12, nhead=8,
        brain_regions=_BCI42A_BRAIN_REGIONS,
        sorted_indices=sorted_indices,
    )

    if os.path.exists(weights_path):
        sd = torch.load(weights_path, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        msd = encoder.state_dict()
        msd.update({k: v for k, v in sd.items()
                    if k in msd and v.size() == msd[k].size()})
        encoder.load_state_dict(msd)
        print(f"Loaded pretrained CSBrain weights from '{weights_path}'.")
    else:
        print(f"WARNING: '{weights_path}' not found — using random init.")

    encoder.proj_out = nn.Identity()  # expose raw 200-dim features

    for p in encoder.parameters():
        p.requires_grad = False

    reducer = EEGTokenReducer(encoder.area_config)
    return encoder, reducer


def pool_eeg(encoder: nn.Module, reducer: EEGTokenReducer,
             eeg: torch.Tensor) -> torch.Tensor:
    """Full EEG → pooled-feature forward pass (no grad).

    Parameters
    ----------
    eeg : (batch, 22, 4, 200)

    Returns
    -------
    pooled : (batch, 200)
    """
    with torch.no_grad():
        feats  = encoder(eeg.float())                         # (B, 22, 4, 200)
        tokens = reducer(feats)                               # (B, n_regions, 4, 200)
        pooled = tokens.reshape(eeg.shape[0], -1, 200).mean(1)  # (B, 200)
    return pooled
