"""
Post-processing for segmentation predictions:
  1. Per-class sigmoid thresholding
  2. Connected component analysis (26-connectivity), min-size filtering
  3. Anatomical constraint: ET ⊆ TC ⊆ WT
"""
from __future__ import annotations
import numpy as np
import cc3d
import torch


THRESHOLDS = {"et": 0.45, "tc": 0.50, "wt": 0.45}
MIN_SIZES = {"et": 50, "tc": 100, "wt": 500}


def filter_small_components(binary: np.ndarray, min_size: int) -> np.ndarray:
    labeled = cc3d.connected_components(binary.astype(np.uint8), connectivity=26)
    out = np.zeros_like(binary)
    for label_id in range(1, labeled.max() + 1):
        component = labeled == label_id
        if component.sum() >= min_size:
            out[component] = 1
    return out


def postprocess(
    pred: torch.Tensor,
    thresholds: dict[str, float] | None = None,
    min_sizes: dict[str, int] | None = None,
) -> np.ndarray:
    """
    Args:
        pred: (3, D, H, W) sigmoid probabilities [ET index 0, TC index 1, WT index 2].

    Returns:
        (3, D, H, W) binary NumPy array after post-processing.
    """
    th = thresholds or THRESHOLDS
    ms = min_sizes or MIN_SIZES

    pred_np = pred.detach().cpu().float().numpy()
    names = ["et", "tc", "wt"]

    binary = np.zeros_like(pred_np, dtype=np.uint8)
    for i, name in enumerate(names):
        b = (pred_np[i] > th[name]).astype(np.uint8)
        binary[i] = filter_small_components(b, ms[name])

    # Anatomical constraints: ET ⊆ TC ⊆ WT
    binary[0] = binary[0] & binary[1]   # ET must be inside TC
    binary[1] = binary[1] & binary[2]   # TC must be inside WT

    return binary
