"""Utilities for loading BraTS JSON fold files and metadata."""
import json
from pathlib import Path
from typing import Any


def load_fold_json(json_path: str | Path, fold: int) -> dict[str, list]:
    """
    Load training and validation lists from a BraTS fold JSON.

    JSON format::

        {
            "training": [
                {
                    "fold": 0,
                    "image": ["t2f.nii.gz", "t1c.nii.gz", "t1n.nii.gz", "t2w.nii.gz"],
                    "label": "seg.nii.gz"
                },
                ...
            ]
        }

    fold -1 entries are always placed in training (synthetic / extra data).
    """
    with open(json_path) as f:
        data = json.load(f)

    train, val = [], []
    for entry in data["training"]:
        if entry["fold"] == fold:
            val.append(entry)
        else:
            train.append(entry)
    return {"train": train, "val": val}


def brats_label_to_channels(label: Any) -> Any:
    """
    Convert single-channel BraTS segmentation to 3-channel binary maps.

    BraTS labels:
        1 = necrotic / non-enhancing core
        2 = peritumoral edema
        4 = enhancing tumor

    Output channels:
        0 (TC)  = labels 1 + 4
        1 (WT)  = labels 1 + 2 + 4
        2 (ET)  = label 4
    """
    import numpy as np
    label = np.array(label)
    tc = ((label == 1) | (label == 4)).astype(np.float32)
    wt = ((label == 1) | (label == 2) | (label == 4)).astype(np.float32)
    et = (label == 4).astype(np.float32)
    return np.stack([tc, wt, et], axis=0)
