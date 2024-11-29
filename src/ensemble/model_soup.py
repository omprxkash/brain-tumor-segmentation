"""
Model Souping: uniform and greedy weight averaging.
Based on: Wortsman et al., Model soups, ICML 2022.

Usage (greedy)::

    python src/ensemble/model_soup.py \\
        --checkpoints checkpoints/fold0/ \\
        --fold 0 \\
        --json brats_fold.json \\
        --npy_dir /data/processed \\
        --output checkpoints/fold0/soup.pt
"""
from __future__ import annotations
import argparse
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric

from src.models import SAUNet3D
from src.training.lightning_module import SegmentationModule
from src.data import BraTSDataset, get_val_transforms
from src.data.dataset_utils import load_fold_json


def _load_state(ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        sd = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        return sd
    return ckpt


def _average_weights(sd_a: dict, sd_b: dict) -> dict:
    return {k: (sd_a[k] + sd_b[k]) / 2.0 for k in sd_a}


def _eval_dice(model: torch.nn.Module, val_dl: DataLoader, device: torch.device) -> float:
    metric = DiceMetric(include_background=True, reduction="mean")
    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            pred = model(x)
            if isinstance(pred, list):
                pred = pred[0]
            metric((pred > 0.5).float(), y)
    result = metric.aggregate().mean().item()
    metric.reset()
    return result


def uniform_soup(ckpt_paths: list[str], device: torch.device) -> dict:
    soup = _load_state(ckpt_paths[0], device)
    for path in ckpt_paths[1:]:
        sd = _load_state(path, device)
        soup = {k: soup[k] + sd[k] for k in soup}
    n = len(ckpt_paths)
    return {k: v / n for k, v in soup.items()}


def greedy_soup(
    ckpt_paths: list[str],
    val_dl: DataLoader,
    base_model: torch.nn.Module,
    device: torch.device,
) -> dict:
    """
    Greedy model souping: start with the best checkpoint,
    greedily add those that improve validation Dice.
    """
    # Sort checkpoints by individual val Dice (descending)
    scored = []
    for path in ckpt_paths:
        sd = _load_state(path, device)
        base_model.load_state_dict(sd)
        score = _eval_dice(base_model, val_dl, device)
        scored.append((score, path, sd))
        print(f"  {Path(path).name}  dice={score:.4f}")
    scored.sort(key=lambda x: x[0], reverse=True)

    soup_sd = scored[0][2]
    soup_score = scored[0][0]
    print(f"Initialising soup with {Path(scored[0][1]).name}  dice={soup_score:.4f}")

    for score, path, sd in scored[1:]:
        candidate = _average_weights(soup_sd, sd)
        base_model.load_state_dict(candidate)
        candidate_score = _eval_dice(base_model, val_dl, device)
        if candidate_score >= soup_score:
            soup_sd = candidate
            soup_score = candidate_score
            print(f"  + Added {Path(path).name}  soup_dice={soup_score:.4f}")
        else:
            print(f"  - Skipped {Path(path).name}  (dice={candidate_score:.4f} < {soup_score:.4f})")

    return soup_sd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True, help="Directory of .ckpt files")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--json", required=True)
    p.add_argument("--npy_dir", required=True)
    p.add_argument("--output", default="checkpoints/soup.pt")
    p.add_argument("--mode", choices=["greedy", "uniform"], default="greedy")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_paths = sorted(Path(args.checkpoints).glob("*.ckpt"))
    if not ckpt_paths:
        raise FileNotFoundError(f"No .ckpt files found in {args.checkpoints}")

    splits = load_fold_json(args.json, args.fold)
    val_ds = BraTSDataset(splits["val"], args.npy_dir, get_val_transforms())
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=2)

    base_model = SAUNet3D().to(device)

    if args.mode == "greedy":
        soup_sd = greedy_soup([str(p) for p in ckpt_paths], val_dl, base_model, device)
    else:
        soup_sd = uniform_soup([str(p) for p in ckpt_paths], device)

    torch.save(soup_sd, args.output)
    print(f"\nSouped model saved → {args.output}")


if __name__ == "__main__":
    main()
