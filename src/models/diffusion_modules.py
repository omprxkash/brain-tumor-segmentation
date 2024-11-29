"""
Core building blocks for the 3D DDPM UNet:
timestep embedding, residual blocks, multi-head QKV attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([args.cos(), args.sin()], dim=-1)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_ch, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(t))[:, :, None, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(self.drop(h))
        return h + self.skip(x)


class AttentionBlock3D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, D * H * W)
        q, k, v = qkv.unbind(1)
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax((q * scale) @ k.transpose(-2, -1), dim=-1)
        out = (attn @ v).reshape(B, C, D, H, W)
        return x + self.proj(out)


class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))
