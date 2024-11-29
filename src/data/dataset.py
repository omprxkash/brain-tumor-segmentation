"""BraTS dataset classes loading preprocessed .npy files from JSON fold config."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose

from .dataset_utils import load_fold_json
from .transforms import get_train_transforms, get_val_transforms


class BraTSDataset(Dataset):
    """
    Loads preprocessed BraTS .npy pairs ({name}_x.npy, {name}_y.npy).

    Args:
        entries:    List of fold-JSON entries (dicts with 'image' and 'label').
        npy_dir:    Directory containing pre-processed .npy files.
        transform:  MONAI Compose transform applied to {"image": x, "label": y}.
    """

    def __init__(self, entries: list[dict], npy_dir: str | Path, transform: Compose | None = None):
        self.entries = entries
        self.npy_dir = Path(npy_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.entries[idx]
        name = Path(entry["label"]).parent.name

        x = np.load(self.npy_dir / f"{name}_x.npy")   # (4, D, H, W)
        y = np.load(self.npy_dir / f"{name}_y.npy")   # (3, D, H, W)

        sample = {"image": x, "label": y}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def build_dataloaders(
    json_path: str,
    npy_dir: str,
    fold: int,
    batch_size: int = 2,
    aug_type: int = 3,
    roi: tuple[int, int, int] = (128, 128, 128),
    num_workers: int = 4,
    pin_memory: bool = True,
):
    splits = load_fold_json(json_path, fold)

    train_ds = BraTSDataset(
        splits["train"], npy_dir, get_train_transforms(aug_type, roi)
    )
    val_ds = BraTSDataset(
        splits["val"], npy_dir, get_val_transforms(roi)
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_dl, val_dl
