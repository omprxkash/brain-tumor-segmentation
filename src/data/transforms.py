"""
3D data augmentation transforms for BraTS segmentation.

Three augmentation levels:
  1 - standard  (flips + mild intensity)
  2 - medium    (affine + moderate intensity)
  3 - aggressive (full affine + strong intensity, default)
"""
from __future__ import annotations
import numpy as np
import torch
from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    SpatialCropd,
    RandSpatialCropd,
    NormalizeIntensityd,
    ToTensord,
    CenterSpatialCropd,
)


KEYS = ["image", "label"]
IMAGE_KEY = ["image"]


def get_train_transforms(aug_type: int = 3, roi: tuple[int, int, int] = (128, 128, 128)):
    base = [RandSpatialCropd(keys=KEYS, roi_size=roi, random_size=False)]

    if aug_type == 1:
        aug = [
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=0),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=1),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=IMAGE_KEY, factors=0.1, prob=1.0),
            RandShiftIntensityd(keys=IMAGE_KEY, offsets=0.1, prob=1.0),
        ]
    elif aug_type == 2:
        aug = [
            RandAffined(
                keys=KEYS, prob=0.7, mode=("bilinear", "nearest"),
                rotate_range=(np.pi / 12,) * 3,
                scale_range=(0.15,) * 3,
            ),
            RandScaleIntensityd(keys=IMAGE_KEY, factors=0.15, prob=0.7),
            RandAdjustContrastd(keys=IMAGE_KEY, prob=0.7, gamma=(0.5, 2.0)),
        ]
    else:  # aug_type == 3 (aggressive)
        aug = [
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=0),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=1),
            RandFlipd(keys=KEYS, prob=0.5, spatial_axis=2),
            RandAffined(
                keys=KEYS, prob=1.0, mode=("bilinear", "nearest"),
                rotate_range=(np.pi / 12,) * 3,
                scale_range=(0.20,) * 3,
                shear_range=(0.20,) * 6,
            ),
            RandScaleIntensityd(keys=IMAGE_KEY, factors=0.15, prob=1.0),
            RandAdjustContrastd(keys=IMAGE_KEY, prob=1.0, gamma=(0.5, 1.5)),
            RandShiftIntensityd(keys=IMAGE_KEY, offsets=0.1, prob=1.0),
        ]

    return Compose(base + aug + [ToTensord(keys=KEYS)])


def get_val_transforms(roi: tuple[int, int, int] = (128, 128, 128)):
    return Compose([ToTensord(keys=KEYS)])
