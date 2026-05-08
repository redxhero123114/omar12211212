# LoLI-Street Low-Light Image Enhancement

> A full deep-learning pipeline for enhancing low-light street images using **CLAHE preprocessing** + a **U-Net** model, trained on the [LoLI-Street dataset](https://www.kaggle.com/datasets/tanvirnwu/loli-street-low-light-image-enhancement-of-street).

---

## Table of Contents

- [Overview](#overview)
- [Problem Definition](#problem-definition)
- [Solution](#solution)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Configuration](#configuration)
- [Results](#results)

---

## Overview

Low-light conditions severely degrade street image quality — reducing visibility, suppressing colour, and introducing noise. This project builds an end-to-end image enhancement pipeline that:

1. **Preprocesses** raw low-light images with CLAHE contrast enhancement and Canny edge detection.
2. **Trains** a U-Net encoder–decoder to learn the mapping from dark input → bright enhanced output.
3. **Evaluates** quality using PSNR and SSIM metrics.

---

## Problem Definition

Street surveillance and autonomous driving systems capture images in varied lighting. At night or in poorly lit environments:

- Pixel intensities are concentrated in dark ranges → low contrast
- Sensor noise becomes dominant
- Edges and textures critical for object detection are suppressed
- Colour information is degraded

Existing histogram equalisation methods over-amplify noise. Deep-learning approaches that learn the enhancement mapping directly from paired data produce significantly better results.

---

## Solution

### Stage 1 — Classical Preprocessing (CLAHE)

Before any model is involved, each image goes through:

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | Resize to 256×256 | Uniform input size |
| 2 | Gaussian Blur (5×5) | Noise suppression |
| 3 | RGB → LAB conversion | Separate luminance from colour |
| 4 | CLAHE on L channel | Adaptive contrast enhancement |
| 5 | LAB → RGB | Restore colour space |
| 6 | Canny Edge Detection | Extract structural features |
| 7 | Normalise to [0, 1] | Stable training |

### Stage 2 — U-Net Deep Learning Model

A **U-Net** architecture is trained end-to-end:

- **Encoder**: 4 blocks of Conv2D + BatchNorm + ReLU + MaxPool (32 → 64 → 128 → 256 filters)
- **Bottleneck**: 512-filter conv block with dropout
- **Decoder**: 4 symmetric upsampling blocks with skip connections from encoder
- **Output**: 1×1 Conv + Sigmoid → enhanced RGB image

### Loss Function

```
L = 0.7 × MAE + 0.3 × (1 − SSIM)
```

This balances pixel-level accuracy (MAE) with perceptual structural quality (SSIM).

---

## Pipeline Architecture

```
Raw Low-Light Image
        │
        ▼
┌─────────────────────┐
│  Preprocessing      │
│  Resize → Denoise   │
│  CLAHE → Edge Map   │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   U-Net Encoder     │
│  32→64→128→256→512  │
└────────┬────────────┘
         │  skip connections
         ▼
┌─────────────────────┐
│   U-Net Decoder     │
│  256→128→64→32      │
└────────┬────────────┘
         │
         ▼
  Enhanced Image (256×256×3)
```

---

## Project Structure

```
.
├── loli_street_enhancement.py   # Full pipeline (data → train → evaluate)
├── README.md                    # This file
├── best_enhancement_model.keras # Saved after training (auto-generated)
├── training_history.png         # Loss/MAE curves (auto-generated)
└── predictions_sample.png       # Visual comparison grid (auto-generated)
```

---

## Setup & Installation

### 1. Install dependencies

```bash
pip install kagglehub opencv-python numpy matplotlib tensorflow
```

### 2. Configure Kaggle credentials (for dataset download)

```bash
# Place kaggle.json in ~/.kaggle/
# Or set environment variables:
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### 3. Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KevCBrCaXpccK0dzMDoEraZxyVod5ctP?usp=sharing)

---

## Usage

### Run the full pipeline

```bash
python loli_street_enhancement.py
```

This will:
1. Show a 3-image preprocessing preview
2. Load up to 7,000 training images
3. Train the U-Net for up to 50 epochs (early stopping enabled)
4. Plot and save training curves
5. Print PSNR / SSIM metrics
6. Show and save 5 sample predictions

### Preprocessing preview only (no training)

```python
from loli_street_enhancement import preview_preprocessing, DATASET_PATH
preview_preprocessing(DATASET_PATH, n=5)
```

### Enhance a single image

```python
import cv2
from loli_street_enhancement import enhance_image

img = cv2.imread("my_dark_image.jpg")
enhanced, edges = enhance_image(img)
```

### Load a saved model and predict

```python
import tensorflow as tf
import numpy as np, cv2
from loli_street_enhancement import enhance_image, IMAGE_SIZE

model = tf.keras.models.load_model("best_enhancement_model.keras",
                                   custom_objects={"combined_loss": combined_loss})

img  = cv2.imread("input.jpg")
x, _ = enhance_image(img)
pred = model.predict(x[None])[0]   # shape (256,256,3)
```

---

## Outputs

| File | Description |
|------|-------------|
| `best_enhancement_model.keras` | Best checkpoint saved during training |
| `training_history.png` | Loss and MAE curves across epochs |
| `predictions_sample.png` | 5-row grid: Input │ Prediction │ Ground Truth |

---

## Configuration

All key hyper-parameters are at the top of `loli_street_enhancement.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | `(256, 256)` | Resize target for all images |
| `MAX_IMAGES` | `7000` | Maximum training samples |
| `BATCH_SIZE` | `16` | Mini-batch size |
| `EPOCHS` | `50` | Max training epochs |
| `LEARNING_RATE` | `1e-4` | Adam optimiser LR |
| `MODEL_SAVE_PATH` | `best_enhancement_model.keras` | Checkpoint filename |
| Canny low threshold | `50` | Edge sensitivity (lower = more edges) |
| Canny high threshold | `150` | Edge sensitivity (higher = fewer edges) |
| CLAHE clip limit | `2.0` | Max contrast amplification |
| CLAHE tile grid | `(8, 8)` | Local region size for CLAHE |

---

## Results

| Metric | Value (expected range) |
|--------|------------------------|
| PSNR | ~22–28 dB |
| SSIM | ~0.75–0.90 |
| Val Loss | Monitored with early stopping |

> Actual values depend on dataset split and hardware. GPU training is strongly recommended.

---

## License

Dataset: [LoLI-Street on Kaggle](https://www.kaggle.com/datasets/tanvirnwu/loli-street-low-light-image-enhancement-of-street) — refer to Kaggle dataset licence.  
Code: MIT
