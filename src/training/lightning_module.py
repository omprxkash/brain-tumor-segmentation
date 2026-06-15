"""
PyTorch Lightning module for brain tumor segmentation.
Supports SA-UNet3D and MedNeXt, schedule-free AdamW,
Dice-Focal loss, optional deep supervision.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
import schedulefree


class SegmentationModule(pl.LightningModule):
    """
    Args:
        model:        Segmentation model (SAUNet3D or MedNeXt).
        lr:           Learning rate for schedule-free AdamW.
        weight_decay: Weight decay.
        deep_sup:     Whether model returns multi-scale outputs during training.
        ds_weights:   Weights for deep supervision levels [1, 1/2, 1/4, ...].
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 2.7e-3,
        weight_decay: float = 0.0,
        deep_sup: bool = False,
        ds_weights: list[float] | None = None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.deep_sup = deep_sup
        self.ds_weights = ds_weights or [1.0, 0.5, 0.25]

        self.criterion = DiceFocalLoss(
            to_onehot_y=False, sigmoid=False, gamma=2.0, batch=True
        )
        self.dice_metric = DiceMetric(include_background=True, reduction="mean")

        self._optimizer = None  # stored for schedule-free eval mode toggle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, preds, targets: torch.Tensor) -> torch.Tensor:
        if self.deep_sup and isinstance(preds, list):
            loss = 0.0
            for w, pred in zip(self.ds_weights, preds):
                if pred.shape != targets.shape:
                    targets_ds = F.interpolate(
                        targets, size=pred.shape[2:], mode="nearest"
                    )
                else:
                    targets_ds = targets
                loss = loss + w * self.criterion(pred, targets_ds)
            return loss
        return self.criterion(preds, targets)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x, y = batch["image"], batch["label"]
        preds = self.model(x)
        loss = self._compute_loss(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        if self._optimizer is not None:
            self._optimizer.eval()
        x, y = batch["image"], batch["label"]
        preds = self.model(x)
        if isinstance(preds, list):
            preds = preds[0]
        loss = self.criterion(preds, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        preds_bin = (preds > 0.5).float()
        self.dice_metric(preds_bin, y)

        if self._optimizer is not None:
            self._optimizer.train()

    def on_validation_epoch_end(self):
        dice = self.dice_metric.aggregate()
        self.dice_metric.reset()
        mean_dice = dice.mean()
        self.log("val_dice", mean_dice, prog_bar=True, sync_dist=True)
        if dice.numel() >= 3:
            self.log("val_dice_et", dice[0], sync_dist=True)
            self.log("val_dice_tc", dice[1], sync_dist=True)
            self.log("val_dice_wt", dice[2], sync_dist=True)

    def configure_optimizers(self):
        opt = schedulefree.AdamWScheduleFree(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self._optimizer = opt
        opt.train()
        return opt
