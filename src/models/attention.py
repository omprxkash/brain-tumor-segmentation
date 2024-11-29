import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock3D(nn.Module):
    """Structured dropout for 3D volumetric feature maps (Ghiasi et al., NeurIPS 2018)."""

    def __init__(self, block_size: int = 7, keep_prob: float = 0.9):
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        _, _, D, H, W = x.shape
        bs = self.block_size
        gamma = ((1 - self.keep_prob) / (bs ** 3)) * (
            (D * H * W) / ((D - bs + 1) * (H - bs + 1) * (W - bs + 1))
        )
        mask = (torch.rand_like(x[:, :1]) < gamma).float()
        mask = F.max_pool3d(mask, kernel_size=bs, stride=1, padding=bs // 2)
        mask = 1.0 - mask
        x = x * mask * (mask.numel() / mask.sum().clamp(min=1.0))
        return x


class SpatialAttention3D(nn.Module):
    """3D spatial attention via avg+max channel pooling → conv → sigmoid (CBAM-style)."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class ChannelAttention3D(nn.Module):
    """Squeeze-and-excitation style channel attention for 3D feature maps."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).view(x.shape[0], x.shape[1], 1, 1, 1)
        return x * w
