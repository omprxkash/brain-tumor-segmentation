"""Dice (with empty-label handling) and HD95 metric utilities."""
from __future__ import annotations
import numpy as np
import torch


def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """
    Dice for binary arrays. BraTS edge cases:
      - both empty  → 1.0
      - pred non-empty, gt empty → 0.0
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not gt.any():
        return 1.0 if not pred.any() else 0.0
    tp = (pred & gt).sum()
    return float(2 * tp / (pred.sum() + gt.sum() + eps))


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile Hausdorff distance (mm). Returns 0 if both empty."""
    from scipy.ndimage import distance_transform_edt
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float("inf")
    dist_pred = distance_transform_edt(~pred)
    dist_gt = distance_transform_edt(~gt)
    hd_a = dist_gt[pred]
    hd_b = dist_pred[gt]
    all_dists = np.concatenate([hd_a, hd_b])
    return float(np.percentile(all_dists, 95))


def compute_metrics(
    pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """
    Compute per-class Dice and HD95 for a single volume.

    Args:
        pred: (3, D, H, W) sigmoid probabilities [ET, TC, WT].
        gt:   (3, D, H, W) binary ground truth.
        threshold: binarization threshold.

    Returns:
        Dict with keys dice_et, dice_tc, dice_wt, hd95_et, hd95_tc, hd95_wt, mean_dice.
    """
    names = ["et", "tc", "wt"]
    pred_np = (pred.detach().cpu().numpy() > threshold)
    gt_np = gt.detach().cpu().numpy().astype(bool)

    results = {}
    dices = []
    for i, name in enumerate(names):
        d = dice_score(pred_np[i], gt_np[i])
        h = hd95(pred_np[i], gt_np[i])
        results[f"dice_{name}"] = d
        results[f"hd95_{name}"] = h
        dices.append(d)
    results["mean_dice"] = float(np.mean(dices))
    return results
