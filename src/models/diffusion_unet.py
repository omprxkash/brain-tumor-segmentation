"""
Conditional 3D UNet backbone for DDPM-based MRI synthesis.
Accepts 8-channel input: 4 noised MRI channels + 4 condition mask channels.
Predicts noise (4-channel output).
"""
import torch
import torch.nn as nn
from .diffusion_modules import (
    timestep_embedding, ResBlock3D, AttentionBlock3D, Downsample3D, Upsample3D,
)


class DiffusionUNet3D(nn.Module):
    """
    3D UNet for conditional DDPM noise prediction.

    Args:
        in_channels:          8 (4 MRI + 4 mask condition channels).
        out_channels:         4 (predicted noise for each MRI modality).
        base_ch:              Base channel count.
        ch_mult:              Channel multipliers per resolution level.
        num_res_blocks:       ResBlocks per level.
        attention_levels:     Which levels (0-indexed) apply self-attention.
        dropout:              Dropout rate in ResBlocks.
        time_embed_dim:       Dimension of sinusoidal timestep embedding.
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 4,
        base_ch: int = 64,
        ch_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_levels: tuple[int, ...] = (2, 3),
        dropout: float = 0.0,
        time_embed_dim: int = 256,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(base_ch, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.in_conv = nn.Conv3d(in_channels, base_ch, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = base_ch
        skip_chs = [ch]
        for lvl, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock3D(ch, out_ch, time_embed_dim, dropout))
                if lvl in attention_levels:
                    self.down_blocks.append(AttentionBlock3D(out_ch))
                skip_chs.append(out_ch)
                ch = out_ch
            if lvl < len(ch_mult) - 1:
                self.down_samples.append(Downsample3D(ch))
                skip_chs.append(ch)
            else:
                self.down_samples.append(nn.Identity())

        # Bottleneck
        self.mid1 = ResBlock3D(ch, ch, time_embed_dim, dropout)
        self.mid_attn = AttentionBlock3D(ch)
        self.mid2 = ResBlock3D(ch, ch, time_embed_dim, dropout)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for lvl, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * mult
            for i in range(num_res_blocks + 1):
                skip_ch = skip_chs.pop()
                self.up_blocks.append(ResBlock3D(ch + skip_ch, out_ch, time_embed_dim, dropout))
                if lvl in attention_levels:
                    self.up_blocks.append(AttentionBlock3D(out_ch))
                ch = out_ch
            if lvl > 0:
                self.up_samples.append(Upsample3D(ch))
            else:
                self.up_samples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv3d(ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(
            timestep_embedding(t, self.time_embed[0].in_features)
        )

        h = self.in_conv(x)
        skips = [h]

        idx_down, idx_ds = 0, 0
        for lvl, _ in enumerate(self.down_samples):
            for _ in range(2):  # num_res_blocks (simplified iteration)
                if idx_down < len(self.down_blocks):
                    block = self.down_blocks[idx_down]
                    if isinstance(block, ResBlock3D):
                        h = block(h, t_emb)
                    else:
                        h = block(h)
                    idx_down += 1
                    skips.append(h)
            ds = self.down_samples[idx_ds]
            idx_ds += 1
            if not isinstance(ds, nn.Identity):
                h = ds(h)
                skips.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, t_emb)), )

        idx_up, idx_us = 0, 0
        for lvl, _ in enumerate(self.up_samples):
            if not isinstance(self.up_samples[idx_us], nn.Identity):
                h = self.up_samples[idx_us](h)
            idx_us += 1
            for _ in range(3):
                if idx_up < len(self.up_blocks):
                    block = self.up_blocks[idx_up]
                    if isinstance(block, ResBlock3D):
                        skip = skips.pop()
                        h = block(torch.cat([h, skip], dim=1), t_emb)
                    else:
                        h = block(h)
                    idx_up += 1

        import torch.nn.functional as F
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h
