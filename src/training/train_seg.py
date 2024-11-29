"""
Segmentation training entry point.

Usage::

    python src/training/train_seg.py --config configs/train_seg.yaml --fold 0
"""
from __future__ import annotations
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.models import SAUNet3D, build_mednext
from src.data import build_dataloaders
from src.training.lightning_module import SegmentationModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_seg.yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def build_model(cfg: dict) -> torch.nn.Module:
    arch = cfg.get("architecture", "sa_unet3d").lower()
    if arch == "sa_unet3d":
        return SAUNet3D(
            in_channels=cfg.get("in_channels", 4),
            out_channels=cfg.get("out_channels", 3),
            base_ch=cfg.get("base_ch", 16),
            block_size=cfg.get("block_size", 7),
            keep_prob=cfg.get("keep_prob", 0.9),
        )
    elif arch.startswith("mednext"):
        size = cfg.get("mednext_size", "B")
        return build_mednext(
            size=size,
            in_channels=cfg.get("in_channels", 4),
            out_channels=cfg.get("out_channels", 3),
            kernel_size=cfg.get("mednext_ksize", 3),
            deep_sup=cfg.get("deep_sup", False),
        )
    raise ValueError(f"Unknown architecture: {arch}")


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fold = args.fold
    max_epochs = args.max_epochs or cfg.get("max_epochs", 100)
    seed = cfg.get("seed", 42)
    pl.seed_everything(seed)

    train_dl, val_dl = build_dataloaders(
        json_path=cfg["json_path"],
        npy_dir=cfg["npy_dir"],
        fold=fold,
        batch_size=cfg.get("batch_size", 2),
        aug_type=cfg.get("aug_type", 3),
        roi=tuple(cfg.get("roi", [128, 128, 128])),
        num_workers=cfg.get("num_workers", 4),
    )

    model = build_model(cfg)
    module = SegmentationModule(
        model=model,
        lr=cfg.get("lr", 2.7e-3),
        weight_decay=cfg.get("weight_decay", 0.0),
        deep_sup=cfg.get("deep_sup", False),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=f"checkpoints/fold{fold}",
            filename="{epoch:03d}-{val_dice:.4f}",
            monitor="val_dice",
            mode="max",
            save_top_k=20,
            save_last=True,
        ),
    ]

    logger = None
    if not args.no_wandb:
        logger = WandbLogger(
            project=cfg.get("wandb_project", "brain-tumor-seg"),
            name=f"fold{fold}_{cfg.get('architecture','sa_unet3d')}",
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.get("n_gpus", 1),
        callbacks=callbacks,
        logger=logger,
        check_val_every_n_epoch=cfg.get("check_val_every_n_epoch", 1),
        precision=32,
    )

    trainer.fit(module, train_dl, val_dl)


if __name__ == "__main__":
    main()
