# CNN-Transformer Video Anomaly Detection (VAD) on UCSD Ped2

This repository contains two implementations of a sequence-based Video Anomaly Detection (VAD) architecture on the **UCSD Ped2** dataset using PyTorch. The overall framework uses a CNN spatial encoder combined with a Transformer temporal decoder to learn normal spatio-temporal patterns and flag deviations as anomalies based on reconstruction/prediction mean squared error (MSE).

---

## 📌 Performance Overview

| Version | Feature Extraction Strategy | Parameters | Global AUC-ROC | Key Features |
| :--- | :--- | :--- | :--- | :--- |
| **v1** | Lightweight Custom ResNet CNN (Trained End-to-End) | ~4.07 M | **0.6295** | Self-Context (50 frames), Anomaly Feedback Loop |
| **v2** | Pretrained ResNet18 (Frozen, Cached to Disk) | ~0.54 M | **0.7123** | Fast Training, Lightweight Transformer-only model |

---

## 🏗️ Architecture & Approach

### 1. Model v1: End-to-End CNN-Transformer with Self-Context & Feedback
* **Spatial Encoder**: Custom ResNet-style 3-block 2D-CNN mapping grayscale frames ($128 \times 128$) to a 256-dimensional feature space.
* **Temporal Model**: 
  * **Encoder**: 2-layer Multi-Head Self-Attention (MHSA) sequence model.
  * **Decoder**: Causal masked Transformer predicting frame $t+1$ given frames $\le t$.
* **Test-Time Adaptation (Self-Context)**: Pre-encodes the first 50 frames of each test clip as additional memory in cross-attention to adapt to clip-specific normality on the fly.
* **Anomaly Feedback Loop**: Replaces the actual feature vector with the model's predicted feature if the frame error exceeds a dynamic anomaly threshold (70th percentile), preventing high-error propagation.

### 2. Model v2: Pretrained Feature Extraction (ResNet18 + Causal Transformer)
* **Spatial Feature Extractor**: Frozen ImageNet-pretrained ResNet18 extracting 512-dimensional feature maps per frame (cached as `.npy` arrays).
* **Temporal Model**:
  * Linear projection from 512 to 128 dimensions.
  * Causal 2-layer Transformer Encoder (`norm_first=True`) trained to predict next-frame features ($t+1$) given prior features ($0..t$).
* **Score Calculation**: Frame-level anomaly score is calculated using the MSE between predicted and actual feature representations.

---

## 📊 Detailed Evaluation Results

### Per-Clip AUC Breakdown (UCSD Ped2)

| Clip | Total Frames | Anomalous Frames | Model v1 AUC | Model v2 AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Test001** | 180 | 120 | 0.7929 | 0.7197 |
| **Test002** | 180 | 86 | 0.4294 | **0.8678** |
| **Test003** | 150 | 146 | **0.9709** | 0.7637 |
| **Test004** | 180 | 150 | **1.0000** | 0.9311 |
| **Test005** | 150 | 129 | 0.6604 | 0.0661 |
| **Test006** | 180 | 159 | 0.1662 | **0.9096** |
| **Test007** | 180 | 135 | 0.6407 | **0.9045** |
| **Test008** | 180 | 180 | *NaN (100% Anomaly)* | *NaN (100% Anomaly)* |
| **Test009** | 120 | 120 | *NaN (100% Anomaly)* | *NaN (100% Anomaly)* |
| **Test010** | 150 | 150 | *NaN (100% Anomaly)* | *NaN (100% Anomaly)* |
| **Test011** | 180 | 180 | *NaN (100% Anomaly)* | *NaN (100% Anomaly)* |
| **Test012** | 180 | 93 | 0.3581 | 0.2703 |
| **Overall** | **2010** | **1648** | **0.6295** | **0.7123** |

*> **Note:** Clips Test008–Test011 contain 100% anomalous frames, causing single-clip ROC-AUC calculations to return `NaN` due to the lack of a ground-truth negative class.*

---

## ⚙️ Configuration Parameters

### Training Setup
```python
# Shared Hyperparameters
IMG_SIZE = 128
SMOOTH_K = 5          # Temporal moving average kernel size
DEVICE   = "cuda"     # Auto-detected PyTorch device

# Model v1 Setup
SEQ_LEN     = 16      # Sequence length
FEAT_DIM    = 256     # Feature dimension
EPOCHS      = 100     # Training epochs
BATCH_SIZE  = 8       # Batch size
LR          = 5e-4    # Learning rate (CosineAnnealingLR)

# Model v2 Setup
SEQ_LEN     = 12
PROJ_DIM    = 128
EPOCHS      = 150
BATCH_SIZE  = 16
LR          = 1e-3

pip install torch torchvision numpy pillow scikit-learn tqdm

UCSDped2/
├── Train/
│   ├── Train001/
│   └── ...
└── Test/
    ├── Test001/
    ├── Test001_gt/
    └── ...

