"""
MedNeXt variants (S/B/M/L) for 3D brain tumor segmentation.
Based on: Roy et al., MedNeXt: Transformer-Driven Scaling of ConvNets
for Medical Image Segmentation, MICCAI 2023.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MedNeXtBlock(nn.Module):
    def __init__(self, in_ch: int, exp_r: int = 4, kernel_size: int = 3, do_res: bool = True):
        super().__init__()
        self.do_res = do_res
        mid = in_ch * exp_r
        pad = kernel_size // 2
        self.conv_dw = nn.Conv3d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False)
        self.norm = nn.LayerNorm(in_ch)
        self.conv_pw1 = nn.Linear(in_ch, mid)
        self.act = nn.GELU()
        self.conv_pw2 = nn.Linear(mid, in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.conv_pw2(self.act(self.conv_pw1(x)))
        x = x.permute(0, 4, 1, 2, 3)
        if self.do_res:
            x = x + residual
        return x


class MedNeXtDownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, exp_r: int = 4, kernel_size: int = 3):
        super().__init__()
        mid = in_ch * exp_r
        pad = kernel_size // 2
        self.conv_dw = nn.Conv3d(in_ch, in_ch, kernel_size, stride=2, padding=pad, groups=in_ch, bias=False)
        self.norm = nn.LayerNorm(in_ch)
        self.conv_pw1 = nn.Linear(in_ch, mid)
        self.act = nn.GELU()
        self.conv_pw2 = nn.Linear(mid, out_ch)
        self.res_conv = nn.Conv3d(in_ch, out_ch, 1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_conv(x)
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.conv_pw2(self.act(self.conv_pw1(x)))
        x = x.permute(0, 4, 1, 2, 3)
        return x + residual


class MedNeXtUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, exp_r: int = 4, kernel_size: int = 3):
        super().__init__()
        mid = in_ch * exp_r
        pad = kernel_size // 2
        self.conv_dw = nn.ConvTranspose3d(
            in_ch, in_ch, kernel_size, stride=2,
            padding=pad, output_padding=1, groups=in_ch, bias=False,
        )
        self.norm = nn.LayerNorm(in_ch)
        self.conv_pw1 = nn.Linear(in_ch, mid)
        self.act = nn.GELU()
        self.conv_pw2 = nn.Linear(mid, out_ch)
        self.res_conv = nn.ConvTranspose3d(in_ch, out_ch, 1, stride=2, output_padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.res_conv(x)
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.conv_pw2(self.act(self.conv_pw1(x)))
        x = x.permute(0, 4, 1, 2, 3)
        return x + residual


def _make_stage(in_ch: int, exp_r: int, n_blocks: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(*[MedNeXtBlock(in_ch, exp_r, kernel_size) for _ in range(n_blocks)])


class MedNeXt(nn.Module):
    """
    MedNeXt segmentation model.

    Args:
        in_channels:  Input modalities (4 for BraTS).
        out_channels: Segmentation classes (3 for ET/TC/WT).
        n_channels:   Base channel count (32 for all variants).
        exp_r:        Expansion ratio per stage (int or list matching 9 stages).
        block_counts: Number of blocks per stage (list of 9).
        kernel_size:  Depth-wise conv kernel size (3 or 5).
        deep_sup:     Enable deep supervision outputs.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        n_channels: int = 32,
        exp_r: int | list = 2,
        block_counts: list[int] | None = None,
        kernel_size: int = 3,
        deep_sup: bool = False,
    ):
        super().__init__()
        if block_counts is None:
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        if isinstance(exp_r, int):
            exp_r = [exp_r] * 9
        c = n_channels
        self.deep_sup = deep_sup

        self.stem = nn.Conv3d(in_channels, c, 1, bias=False)

        self.enc1 = _make_stage(c, exp_r[0], block_counts[0], kernel_size)
        self.down1 = MedNeXtDownBlock(c, c * 2, exp_r[1], kernel_size)

        self.enc2 = _make_stage(c * 2, exp_r[1], block_counts[1], kernel_size)
        self.down2 = MedNeXtDownBlock(c * 2, c * 4, exp_r[2], kernel_size)

        self.enc3 = _make_stage(c * 4, exp_r[2], block_counts[2], kernel_size)
        self.down3 = MedNeXtDownBlock(c * 4, c * 8, exp_r[3], kernel_size)

        self.enc4 = _make_stage(c * 8, exp_r[3], block_counts[3], kernel_size)
        self.down4 = MedNeXtDownBlock(c * 8, c * 16, exp_r[4], kernel_size)

        self.bottleneck = _make_stage(c * 16, exp_r[4], block_counts[4], kernel_size)

        self.up4 = MedNeXtUpBlock(c * 16, c * 8, exp_r[5], kernel_size)
        self.dec4 = _make_stage(c * 8, exp_r[5], block_counts[5], kernel_size)

        self.up3 = MedNeXtUpBlock(c * 8, c * 4, exp_r[6], kernel_size)
        self.dec3 = _make_stage(c * 4, exp_r[6], block_counts[6], kernel_size)

        self.up2 = MedNeXtUpBlock(c * 4, c * 2, exp_r[7], kernel_size)
        self.dec2 = _make_stage(c * 2, exp_r[7], block_counts[7], kernel_size)

        self.up1 = MedNeXtUpBlock(c * 2, c, exp_r[8], kernel_size)
        self.dec1 = _make_stage(c, exp_r[8], block_counts[8], kernel_size)

        self.out = nn.Conv3d(c, out_channels, 1)

        if deep_sup:
            self.ds2 = nn.Conv3d(c * 2, out_channels, 1)
            self.ds3 = nn.Conv3d(c * 4, out_channels, 1)
            self.ds4 = nn.Conv3d(c * 8, out_channels, 1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))

        d4 = self.dec4(self.up4(b) + e4)
        d3 = self.dec3(self.up3(d4) + e3)
        d2 = self.dec2(self.up2(d3) + e2)
        d1 = self.dec1(self.up1(d2) + e1)

        out = torch.sigmoid(self.out(d1))

        if self.deep_sup and self.training:
            return [
                out,
                torch.sigmoid(self.ds2(d2)),
                torch.sigmoid(self.ds3(d3)),
                torch.sigmoid(self.ds4(d4)),
            ]
        return out


# ── Preset constructors ───────────────────────────────────────────────────────

def mednext_s(in_channels=4, out_channels=3, kernel_size=3, deep_sup=False) -> MedNeXt:
    return MedNeXt(in_channels, out_channels, 32, 2, [2]*9, kernel_size, deep_sup)


def mednext_b(in_channels=4, out_channels=3, kernel_size=3, deep_sup=False) -> MedNeXt:
    return MedNeXt(
        in_channels, out_channels, 32,
        [2, 3, 4, 4, 4, 4, 4, 3, 2], [2]*9, kernel_size, deep_sup,
    )


def mednext_m(in_channels=4, out_channels=3, kernel_size=3, deep_sup=False) -> MedNeXt:
    return MedNeXt(
        in_channels, out_channels, 32,
        [2, 3, 4, 4, 4, 4, 4, 3, 2], [3, 4, 4, 4, 4, 4, 4, 4, 3], kernel_size, deep_sup,
    )


def mednext_l(in_channels=4, out_channels=3, kernel_size=3, deep_sup=False) -> MedNeXt:
    return MedNeXt(
        in_channels, out_channels, 32,
        [3, 4, 8, 8, 8, 8, 8, 4, 3], [3, 4, 8, 8, 8, 8, 8, 4, 3], kernel_size, deep_sup,
    )


MODEL_REGISTRY = {"S": mednext_s, "B": mednext_b, "M": mednext_m, "L": mednext_l}


def build_mednext(size: str = "B", **kwargs) -> MedNeXt:
    if size not in MODEL_REGISTRY:
        raise ValueError(f"Unknown MedNeXt size '{size}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[size](**kwargs)
