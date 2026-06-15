"""
Cross-validation evaluation: runs inference on all 5 fold validation sets
and aggregates mean Dice and HD95 across all cases.

Usage::

    python scripts/evaluate_cv.py \\
        --checkpoint_dir checkpoints/ \\
        --json data/brats_ssa_2024_5fold.json \\
        --npy_dir data/processed
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.models import SAUNet3D
from src.data import BraTSDataset, get_val_transforms
from src.data.dataset_utils import load_fold_json
from src.inference.infer_seg import infer
from src.evaluation.metrics import compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--npy_dir", required=True)
    p.add_argument("--n_folds", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []

    for fold in range(args.n_folds):
        soup_path = Path(args.checkpoint_dir) / f"fold{fold}" / "soup.pt"
        best_path = sorted(Path(args.checkpoint_dir) / f"fold{fold}" / "*.ckpt")
        if soup_path.exists():
            ckpt = str(soup_path)
        elif best_path:
            ckpt = str(best_path[-1])
        else:
            print(f"No checkpoint for fold {fold}, skipping.")
            continue

        from src.training.lightning_module import SegmentationModule
        module = SegmentationModule.load_from_checkpoint(
            ckpt, model=SAUNet3D(), map_location=device
        )
        model = module.model.to(device)
        model.eval()

        splits = load_fold_json(args.json, fold)
        val_ds = BraTSDataset(splits["val"], args.npy_dir, get_val_transforms())

        for entry in val_ds.entries:
            from pathlib import Path as P
            name = P(entry["label"]).parent.name
            x = np.load(Path(args.npy_dir) / f"{name}_x.npy")
            y = np.load(Path(args.npy_dir) / f"{name}_y.npy")
            pred = infer(model, x, device=device)
            metrics = compute_metrics(
                torch.from_numpy(pred.astype(np.float32)),
                torch.from_numpy(y),
                threshold=0.5,
            )
            metrics["fold"] = fold
            metrics["case"] = name
            all_results.append(metrics)
            print(f"  fold{fold} {name}  mean_dice={metrics['mean_dice']:.4f}")

    if all_results:
        for key in ["mean_dice", "dice_et", "dice_tc", "dice_wt", "hd95_et", "hd95_tc", "hd95_wt"]:
            vals = [r[key] for r in all_results if r[key] != float("inf")]
            print(f"{key:15s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        with open("cv_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\nFull results saved → cv_results.json")


if __name__ == "__main__":
    main()
