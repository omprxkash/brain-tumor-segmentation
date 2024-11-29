"""
DDPM training for conditional 3D MRI synthesis.
Trains a DiffusionUNet3D to predict noise added at each diffusion timestep.

Usage::

    python src/training/train_ddpm.py --config configs/train_ddpm.yaml
"""
from __future__ import annotations
import argparse
import math
import copy
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path

from src.models import DiffusionUNet3D
from src.data import BraTSDataset, get_val_transforms
from src.data.dataset_utils import load_fold_json


def cosine_beta_schedule(T: int) -> torch.Tensor:
    steps = torch.linspace(0, T, T + 1)
    alpha_bar = torch.cos(((steps / T) + 0.008) / 1.008 * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0, 0.999)


class GaussianDiffusion:
    def __init__(self, betas: torch.Tensor, device: torch.device):
        self.T = len(betas)
        betas = betas.to(device)
        alphas = 1.0 - betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alphas_bar[t][:, None, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def ddim_sample(
        self, model: torch.nn.Module, mask: torch.Tensor, steps: int = 10
    ) -> torch.Tensor:
        B, _, D, H, W = mask.shape
        x = torch.randn(B, 4, D, H, W, device=mask.device)
        indices = torch.linspace(self.T - 1, 0, steps).long().to(mask.device)
        for i, t_val in enumerate(indices):
            t = torch.full((B,), t_val, device=mask.device, dtype=torch.long)
            inp = torch.cat([x, mask], dim=1)
            eps = model(inp, t)
            ab = self.alphas_bar[t_val]
            ab_prev = self.alphas_bar[indices[i + 1]] if i + 1 < len(indices) else torch.tensor(1.0)
            x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
        return x


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    betas = cosine_beta_schedule(cfg["T"])
    diffusion = GaussianDiffusion(betas, device)

    model = DiffusionUNet3D(
        in_channels=8,
        out_channels=4,
        base_ch=cfg.get("base_ch", 64),
    ).to(device)

    ema_model = copy.deepcopy(model)
    ema_decay = cfg.get("ema_decay", 0.995)
    ema_start = cfg.get("ema_start_step", 2000)
    ema_update_every = cfg.get("ema_update_every", 10)

    splits = load_fold_json(cfg["json_path"], fold=0)
    ds = BraTSDataset(splits["train"], cfg["npy_dir"], get_val_transforms())
    dl = DataLoader(ds, batch_size=cfg.get("batch_size", 1), shuffle=True, num_workers=2)

    optimizer = Adam(model.parameters(), lr=cfg.get("lr", 1e-5))
    grad_accum = cfg.get("grad_accum", 2)
    total_steps = cfg.get("total_steps", 100_000)

    out_dir = Path(cfg.get("checkpoint_dir", "checkpoints/ddpm"))
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    optimizer.zero_grad()
    model.train()

    while step < total_steps:
        for batch in dl:
            x0 = batch["image"].to(device)           # (B, 4, D, H, W)
            mask = batch["label"].to(device)          # (B, 3, D, H, W)
            bg = 1 - mask.sum(1, keepdim=True).clamp(0, 1)
            cond = torch.cat([mask, bg], dim=1)       # (B, 4, D, H, W)

            t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            xt = diffusion.q_sample(x0, t, noise)

            inp = torch.cat([xt, cond], dim=1)        # (B, 8, D, H, W)
            pred_noise = model(inp, t)
            loss = F.l1_loss(pred_noise, noise) / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step >= ema_start and step % ema_update_every == 0:
                for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            if (step + 1) % 1000 == 0:
                torch.save(
                    {"model": model.state_dict(), "ema": ema_model.state_dict(), "step": step},
                    out_dir / f"ddpm_step{step+1}.pt",
                )
                print(f"Step {step+1}/{total_steps}  loss={loss.item() * grad_accum:.4f}")

            step += 1
            if step >= total_steps:
                break

    torch.save(ema_model.state_dict(), out_dir / "ddpm_ema_final.pt")
    print("DDPM training complete.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_ddpm.yaml")
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
