"""
BraTS preprocessing pipeline: NIfTI → normalised NumPy arrays.

Usage::

    python scripts/preprocess.py --data_dir /path/to/brats --out_dir /path/to/processed
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


MODALITY_ORDER = ["t2f", "t1c", "t1n", "t2w"]


def load_nifti(path: str | Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance on non-zero voxels, clipped to [-5, 5]."""
    mask = vol != 0
    if mask.sum() == 0:
        return vol
    mu, sigma = vol[mask].mean(), vol[mask].std() + 1e-8
    vol = (vol - mu) / sigma
    return np.clip(vol, -5.0, 5.0)


def process_case(args: tuple) -> None:
    entry, out_dir = args
    images = entry["image"]
    label_path = entry["label"]

    out_dir = Path(out_dir)
    name = Path(label_path).parent.name

    vols = []
    for path in images:
        vol = normalize_volume(load_nifti(path))
        vols.append(vol)

    x = np.stack(vols, axis=0).astype(np.float32)

    label_raw = load_nifti(label_path).astype(np.uint8)
    tc = ((label_raw == 1) | (label_raw == 4)).astype(np.float32)
    wt = ((label_raw == 1) | (label_raw == 2) | (label_raw == 4)).astype(np.float32)
    et = (label_raw == 4).astype(np.float32)
    y = np.stack([tc, wt, et], axis=0)

    np.save(out_dir / f"{name}_x.npy", x)
    np.save(out_dir / f"{name}_y.npy", y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to fold JSON file")
    parser.add_argument("--out_dir", required=True, help="Output directory for .npy files")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = [(entry, str(out)) for entry in data["training"]]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(process_case, tasks))
    print(f"Preprocessed {len(tasks)} cases → {out}")


if __name__ == "__main__":
    main()
