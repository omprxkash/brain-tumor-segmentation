"""
SA-UNet3D: 3D Spatial Attention UNet for brain tumor segmentation.
Architecture: 4-level encoder-decoder, channels 16→32→64→128→256,
SpatialAttention3D at bottleneck, DropBlock3D regularization,
sigmoid 3-channel output (ET, TC, WT).
"""
import torch
import torch.nn as nn
from .attention import DropBlock3D, SpatialAttention3D


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        block_size: int = 7,
        keep_prob: float = 0.9,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            DropBlock3D(block_size, keep_prob),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            DropBlock3D(block_size, keep_prob),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SAUNet3D(nn.Module):
    """
    Spatial Attention 3D UNet.

    Args:
        in_channels:  Number of input modalities (4 for BraTS).
        out_channels: Number of segmentation classes (3 for ET/TC/WT).
        base_ch:      Base channel width (doubles per level).
        block_size:   DropBlock cubic region side length.
        keep_prob:    DropBlock retention probability.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_ch: int = 16,
        block_size: int = 7,
        keep_prob: float = 0.9,
    ):
        super().__init__()
        c = [base_ch * (2 ** i) for i in range(5)]   # 16 32 64 128 256

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, c[0], block_size, keep_prob)
        self.enc2 = ConvBlock3D(c[0], c[1], block_size, keep_prob)
        self.enc3 = ConvBlock3D(c[1], c[2], block_size, keep_prob)
        self.enc4 = ConvBlock3D(c[2], c[3], block_size, keep_prob)

        self.pool = nn.MaxPool3d(2, stride=2)

        # Bottleneck + spatial attention
        self.bottleneck = ConvBlock3D(c[3], c[4], block_size, keep_prob)
        self.attn = SpatialAttention3D(kernel_size=3)

        # Decoder
        self.up4 = nn.ConvTranspose3d(c[4], c[3], 2, stride=2)
        self.dec4 = ConvBlock3D(c[4], c[3], block_size, keep_prob)

        self.up3 = nn.ConvTranspose3d(c[3], c[2], 2, stride=2)
        self.dec3 = ConvBlock3D(c[3], c[2], block_size, keep_prob)

        self.up2 = nn.ConvTranspose3d(c[2], c[1], 2, stride=2)
        self.dec2 = ConvBlock3D(c[2], c[1], block_size, keep_prob)

        self.up1 = nn.ConvTranspose3d(c[1], c[0], 2, stride=2)
        self.dec1 = ConvBlock3D(c[1], c[0], block_size, keep_prob)

        self.out_conv = nn.Conv3d(c[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.attn(self.bottleneck(self.pool(e4)))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))
