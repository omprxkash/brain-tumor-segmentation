"""
DDIM sampling for synthetic MRI generation.
Generates 4-channel MRI volumes conditioned on segmentation masks.

Usage::

    python src/inference/sample_ddpm.py \\
        --checkpoint checkpoints/ddpm/ddpm_ema_final.pt \\
        --mask path/to/mask_y.npy \\
        --output path/to/synthetic_x.npy \\
        --steps 10
"""
from __future__ import annotations
import argparse
import math
import numpy as np
import torch
from src.models import DiffusionUNet3D


def cosine_beta_schedule(T: int) -> torch.Tensor:
    steps = torch.linspace(0, T, T + 1)
    alpha_bar = torch.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0, 0.999)


@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    mask: torch.Tensor,
    alphas_bar: torch.Tensor,
    steps: int = 10,
) -> torch.Tensor:
    """
    Args:
        mask: (1, 4, D, H, W) condition channels (ET, TC, WT, background).

    Returns:
        (1, 4, D, H, W) synthetic MRI volume.
    """
    device = mask.device
    B = mask.shape[0]
    D, H, W = mask.shape[2:]
    x = torch.randn(B, 4, D, H, W, device=device)

    T = len(alphas_bar)
    indices = torch.linspace(T - 1, 0, steps).long()

    for i, t_val in enumerate(indices):
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        inp = torch.cat([x, mask], dim=1)
        eps = model(inp, t)

        ab = alphas_bar[t_val].to(device)
        if i + 1 < len(indices):
            ab_prev = alphas_bar[indices[i + 1]].to(device)
        else:
            ab_prev = torch.tensor(1.0, device=device)

        x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-8)
        x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps

    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mask", required=True, help="Path to _y.npy (3-channel label)")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--T", type=int, default=250)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiffusionUNet3D(in_channels=8, out_channels=4).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    betas = cosine_beta_schedule(args.T)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    label = np.load(args.mask)                     # (3, D, H, W)
    bg = 1 - label.sum(0, keepdims=True).clip(0, 1)
    cond = np.concatenate([label, bg], axis=0)     # (4, D, H, W)
    mask_t = torch.from_numpy(cond).unsqueeze(0).float().to(device)

    synth = ddim_sample(model, mask_t, alphas_bar, steps=args.steps)
    synth_np = synth.squeeze(0).cpu().numpy()      # (4, D, H, W)

    # Normalize to [0, 1] per modality
    for i in range(4):
        mn, mx = synth_np[i].min(), synth_np[i].max()
        synth_np[i] = (synth_np[i] - mn) / (mx - mn + 1e-8)

    np.save(args.output, synth_np.astype(np.float32))
    print(f"Saved synthetic MRI → {args.output}")


if __name__ == "__main__":
    main()
