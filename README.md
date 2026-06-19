# Brain Tumor Segmentation

**Enhancing 3D Brain Tumor Segmentation via Spatial Attention Networks, Schedule-Free Optimization, Diffusion-Based Augmentation, and Greedy Model Souping**

*Omprakash Pugazhendhi — Dept. of Computer Science and Engineering, Vellore Institute of Technology, Chennai, India*

---

## Paper

Read the full research paper here: [`paper/brain_tumor_segmentation.pdf`](paper/brain_tumor_segmentation.pdf)

---

## Overview

This framework targets the BraTS 2024 Sub-Saharan Africa challenge, combining four techniques into a single end-to-end pipeline:

| Component | What it does |
|-----------|-------------|
| **SA-UNet3D** | 3D encoder-decoder with spatial attention at the bottleneck and DropBlock regularization — focuses the network on tumor sub-regions |
| **Schedule-Free AdamW** | Replaces cosine/polynomial LR schedules with a single optimizer that needs no schedule tuning |
| **Conditional 3D DDPM** | Generates synthetic multimodal MRI volumes from segmentation masks, expanding training data by ~30% per fold |
| **Greedy Model Souping** | Iteratively averages checkpoint weights, keeping only improvements — better generalization at zero inference cost |

### Results on BraTS 2024 SSA (5-Fold CV)

| Configuration | ET Dice | TC Dice | WT Dice | ET HD95 | TC HD95 | WT HD95 |
|---------------|---------|---------|---------|---------|---------|---------|
| Baseline UNet | 0.801 | 0.841 | 0.873 | 18.4 | 14.2 | 8.7 |
| + Spatial Attention | 0.817 | 0.856 | 0.885 | 16.2 | 12.8 | 7.9 |
| + Schedule-Free | 0.826 | 0.864 | 0.891 | 15.1 | 11.9 | 7.4 |
| + DDPM Augmentation | 0.839 | 0.873 | 0.899 | 13.7 | 10.8 | 6.8 |
| **Ours (full)** | **0.847** | **0.879** | **0.903** | **12.9** | **10.2** | **6.3** |

---

## Repository Structure

```
brain-tumor-segmentation/
├── paper/
│   ├── brain_tumor_segmentation.pdf   ← compiled IEEE paper
│   └── paper.tex                      ← LaTeX source
├── src/
│   ├── models/
│   │   ├── sa_unet_3d.py              ← SA-UNet3D (primary model)
│   │   ├── mednext.py                 ← MedNeXt S/B/M/L
│   │   ├── attention.py               ← SpatialAttention3D, DropBlock3D
│   │   ├── diffusion_unet.py          ← DDPM UNet backbone
│   │   └── diffusion_modules.py       ← ResBlock, QKV attention, timestep embedding
│   ├── data/
│   │   ├── dataset.py                 ← BraTS dataset loader
│   │   ├── transforms.py              ← 3-level augmentation pipeline
│   │   ├── preprocessing.py           ← NIfTI → NumPy normalization
│   │   └── dataset_utils.py           ← fold JSON parsing, label conversion
│   ├── training/
│   │   ├── lightning_module.py        ← PyTorch Lightning module
│   │   ├── train_seg.py               ← segmentation training entry point
│   │   └── train_ddpm.py              ← diffusion model training
│   ├── inference/
│   │   ├── infer_seg.py               ← sliding-window inference
│   │   └── sample_ddpm.py             ← DDIM 10-step MRI synthesis
│   ├── ensemble/
│   │   └── model_soup.py              ← uniform + greedy souping
│   └── evaluation/
│       ├── metrics.py                 ← Dice, HD95
│       └── postprocessing.py          ← CCA, thresholds, ET⊆TC⊆WT
├── configs/
│   ├── train_seg.yaml                 ← segmentation hyperparameters
│   └── train_ddpm.yaml                ← diffusion hyperparameters
├── scripts/
│   ├── preprocess.py                  ← batch NIfTI preprocessing
│   └── evaluate_cv.py                 ← 5-fold evaluation aggregation
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Key dependencies:** PyTorch 2.0+, MONAI 1.1, PyTorch Lightning 1.9, schedulefree, nibabel, cc3d

---

## Usage

### 1. Preprocess BraTS data

Create a fold JSON (`data/brats_ssa_2024_5fold.json`) following BraTS format, then:

```bash
python scripts/preprocess.py \
    --json data/brats_ssa_2024_5fold.json \
    --out_dir data/processed \
    --workers 8
```

### 2. Train segmentation model (all 5 folds)

```bash
for fold in 0 1 2 3 4; do
  python src/training/train_seg.py --config configs/train_seg.yaml --fold $fold
done
```

### 3. Train diffusion model (for synthetic augmentation)

```bash
python src/training/train_ddpm.py --config configs/train_ddpm.yaml
```

### 4. Generate synthetic volumes

```bash
python src/inference/sample_ddpm.py \
    --checkpoint checkpoints/ddpm/ddpm_ema_final.pt \
    --mask data/processed/case001_y.npy \
    --output data/synthetic/case001_x.npy \
    --steps 10
```

### 5. Greedy model souping (per fold)

```bash
python src/ensemble/model_soup.py \
    --checkpoints checkpoints/fold0/ \
    --fold 0 \
    --json data/brats_ssa_2024_5fold.json \
    --npy_dir data/processed \
    --output checkpoints/fold0/soup.pt
```

### 6. Inference

```bash
python src/inference/infer_seg.py \
    --checkpoint checkpoints/fold0/soup.pt \
    --input data/processed/case001_x.npy \
    --output predictions/case001_pred.npy
```

### 7. Cross-validation evaluation

```bash
python scripts/evaluate_cv.py \
    --checkpoint_dir checkpoints/ \
    --json data/brats_ssa_2024_5fold.json \
    --npy_dir data/processed
```

---

## Citation

If you use this work, please cite:

```
O. Pugazhendhi, "Enhancing 3D Brain Tumor Segmentation via Spatial Attention
Networks, Schedule-Free Optimization, Diffusion-Based Augmentation, and Greedy
Model Souping," IEEE Conference, 2024.
```
