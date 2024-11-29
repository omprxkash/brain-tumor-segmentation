"""
Sliding-window inference for segmentation.

Usage::

    python src/inference/infer_seg.py \\
        --checkpoint checkpoints/fold0/best.ckpt \\
        --input path/to/case_x.npy \\
        --output path/to/pred.npy
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from src.models import SAUNet3D
from src.training.lightning_module import SegmentationModule
from src.evaluation.postprocessing import postprocess


def load_model(checkpoint: str, device: torch.device) -> torch.nn.Module:
    module = SegmentationModule.load_from_checkpoint(
        checkpoint, model=SAUNet3D(), map_location=device
    )
    module.eval()
    if module._optimizer is not None:
        module._optimizer.eval()
    return module.model.to(device)


def infer(
    model: torch.nn.Module,
    x: np.ndarray,
    roi: tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    device: torch.device | None = None,
) -> np.ndarray:
    if device is None:
        device = next(model.parameters()).device
    inp = torch.from_numpy(x).unsqueeze(0).to(device)   # (1, 4, D, H, W)
    with torch.no_grad():
        pred = sliding_window_inference(
            inp, roi, sw_batch_size, model, overlap=overlap
        )
    return postprocess(pred[0])  # (3, D, H, W)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="Path to _x.npy file")
    p.add_argument("--output", required=True, help="Output .npy path")
    p.add_argument("--roi", nargs=3, type=int, default=[128, 128, 128])
    p.add_argument("--overlap", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    x = np.load(args.input)
    pred = infer(model, x, roi=tuple(args.roi), overlap=args.overlap, device=device)
    np.save(args.output, pred)
    print(f"Saved prediction → {args.output}  shape={pred.shape}")


if __name__ == "__main__":
    main()
