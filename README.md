# Brain Tumor Segmentation

3D brain tumor segmentation framework for the BraTS 2024 challenge,
combining spatial attention UNets, schedule-free optimization,
diffusion-based data augmentation, and greedy model souping.

## Overview

| Component | Description |
|-----------|-------------|
| **SA-UNet3D** | 3D encoder-decoder with spatial attention at bottleneck and DropBlock regularization |
| **MedNeXt** | Large-kernel ConvNeXt-style 3D segmentation network (S/B/M/L variants) |
| **Schedule-Free AdamW** | Optimizer without learning rate schedule (Defazio et al., NeurIPS 2024) |
| **DDPM Augmentation** | Conditional 3D diffusion model for synthetic MRI generation |
| **Greedy Souping** | Selective checkpoint weight averaging (Wortsman et al., ICML 2022) |

## Installation

```bash
pip install -e .
pip install -r requirements.txt
```

## Data Preparation

1. Download BraTS 2024 SSA dataset and create a fold JSON:
   ```
   data/brats_ssa_2024_5fold.json
   ```

2. Preprocess NIfTI volumes to NumPy arrays:
   ```bash
   python scripts/preprocess.py \
       --json data/brats_ssa_2024_5fold.json \
       --out_dir data/processed \
       --workers 8
   ```

## Training

### Segmentation (SA-UNet3D, all 5 folds)

```bash
for fold in 0 1 2 3 4; do
  python src/training/train_seg.py \
      --config configs/train_seg.yaml \
      --fold $fold
done
```

### Diffusion Model (for synthetic augmentation)

```bash
python src/training/train_ddpm.py --config configs/train_ddpm.yaml
```

### Generate Synthetic Volumes

```bash
python src/inference/sample_ddpm.py \
    --checkpoint checkpoints/ddpm/ddpm_ema_final.pt \
    --mask data/processed/case001_y.npy \
    --output data/synthetic/case001_synthetic_x.npy \
    --steps 10
```

### Greedy Model Souping (per fold)

```bash
python src/ensemble/model_soup.py \
    --checkpoints checkpoints/fold0/ \
    --fold 0 \
    --json data/brats_ssa_2024_5fold.json \
    --npy_dir data/processed \
    --output checkpoints/fold0/soup.pt \
    --mode greedy
```

## Inference

```bash
python src/inference/infer_seg.py \
    --checkpoint checkpoints/fold0/soup.pt \
    --input data/processed/case001_x.npy \
    --output predictions/case001_pred.npy
```

## Evaluation (5-Fold Cross-Validation)

```bash
python scripts/evaluate_cv.py \
    --checkpoint_dir checkpoints/ \
    --json data/brats_ssa_2024_5fold.json \
    --npy_dir data/processed
```

## Project Structure

```
brain-tumor-segmentation/
├── src/
│   ├── models/          # SA-UNet3D, MedNeXt, attention modules, DDPM UNet
│   ├── data/            # Dataset, transforms, preprocessing
│   ├── training/        # Lightning module, train_seg, train_ddpm
│   ├── inference/       # Sliding-window inference, DDIM sampling
│   ├── ensemble/        # Uniform and greedy model souping
│   └── evaluation/      # Dice, HD95, post-processing
├── configs/             # YAML config files
├── scripts/             # Preprocessing and evaluation scripts
├── paper/               # IEEE LaTeX paper
└── requirements.txt
```

## Results (BraTS 2024 SSA, 5-Fold CV)

| Method | ET Dice | TC Dice | WT Dice | ET HD95 | TC HD95 | WT HD95 |
|--------|---------|---------|---------|---------|---------|---------|
| Baseline UNet | 0.801 | 0.841 | 0.873 | 18.4 | 14.2 | 8.7 |
| + Spatial Attention | 0.817 | 0.856 | 0.885 | 16.2 | 12.8 | 7.9 |
| + Schedule-Free | 0.826 | 0.864 | 0.891 | 15.1 | 11.9 | 7.4 |
| + DDPM Augmentation | 0.839 | 0.873 | 0.899 | 13.7 | 10.8 | 6.8 |
| **+ Model Souping (Ours)** | **0.847** | **0.879** | **0.903** | **12.9** | **10.2** | **6.3** |

## Citation

If you use this work, please cite the associated paper (see `paper/paper.tex`).
