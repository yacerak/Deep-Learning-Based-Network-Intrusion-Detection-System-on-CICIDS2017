# **Deep Learning–Based Network Intrusion Detection System on CICIDS-2017**

> A progressive, multi-architecture framework for network intrusion detection combining Variational Autoencoders (VAE), LightGBM, a hierarchical gated ensemble, and CatBoost–MLP knowledge distillation on the CICIDS-2017 benchmark dataset.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure & Notebook Descriptions](#2-repository-structure--notebook-descriptions)
3. [Dataset: CICIDS-2017](#3-dataset-cicids-2017)
4. [Data Preprocessing Pipeline](#4-data-preprocessing-pipeline)
5. [Model Architectures](#5-model-architectures)
   - 5.1 [Variational Autoencoder (Anomaly Detection)](#51-variational-autoencoder-anomaly-detection)
   - 5.2 [VAE + LightGBM Gated System](#52-vae--lightgbm-gated-system)
   - 5.3 [Hierarchical VAE–LightGBM Gated System](#53-hierarchical-vaelightgbm-gated-system)
   - 5.4 [CatBoost Teacher + MLP Student Distillation](#54-catboost-teacher--mlp-student-distillation)
6. [Training Strategy & Hyperparameters](#6-training-strategy--hyperparameters)
7. [Evaluation & Results Discussion](#7-evaluation--results-discussion)
8. [Dependencies & Environment](#8-dependencies--environment)
9. [Reproduction Guide](#9-reproduction-guide)
10. [Limitations & Future Work](#10-limitations--future-work)

---

## 1. Project Overview

Network intrusion detection remains one of the most critical and technically demanding problems in cybersecurity. Classical signature-based systems fail against zero-day threats and novel attack variants, motivating the shift toward data-driven, learning-based approaches. This project constructs a comprehensive, layered IDS framework that evolves across five progressive notebooks — from raw data engineering to state-of-the-art knowledge distillation — all evaluated on the widely adopted **CICIDS-2017** benchmark.

The project addresses three core challenges in IDS research:

- **Severe class imbalance**: benign traffic dominates by orders of magnitude over rare attack categories such as Infiltration and Web Attacks.
- **Multi-class vs. binary detection**: distinguishing not just malicious from benign, but also precisely identifying the attack category.
- **Inference efficiency**: deploying a lightweight model that retains near-teacher accuracy, suitable for edge or real-time deployment.

The solution progressively stacks: a preprocessing pipeline → an unsupervised VAE anomaly detector → a two-stage VAE-gated LightGBM classifier → a hierarchical per-class VAE ensemble with a learned meta-gate → and finally a CatBoost teacher distilled into a shallow MLP student with confidence-based routing.

---

## 2. Repository Structure & Notebook Descriptions

The five notebooks form a sequential pipeline where each stage builds on artifacts produced by the previous one. The recommended rename mapping is:

| Filename | Role |
|---|---|
| `cicids2017_data_processing.ipynb` | Data ingestion, EDA, feature engineering, preprocessing, artifact export |
| `vae_anomaly_detection.ipynb` | Unsupervised VAE trained on benign-only traffic; binary anomaly detection via reconstruction error |
| `vae_lgbm_gated_classifier.ipynb` | Two-stage system: VAE gate filters benign traffic, LightGBM handles multi-class attack classification |
| `hierarchical_vae_lgbm_gated_ids.ipynb` | Per-class VAE ensemble with a learned meta-gate combining VAE reconstruction and LightGBM posterior |
| `catboost_mlp_knowledge_distillation.ipynb` | CatBoost teacher trained at full precision; shallow MLP student trained via KL-divergence distillation with confidence-based routing |

```
ids-cicids2017/
├── cicids2017_data_processing.ipynb
├── vae_anomaly_detection.ipynb
├── vae_lgbm_gated_classifier.ipynb
├── hierarchical_vae_lgbm_gated_ids.ipynb
├── catboost_mlp_knowledge_distillation.ipynb
└── README.md
```

Each notebook exports serialized artifacts (scalers, encoders, models, configuration JSON) consumed by downstream notebooks, forming a reproducible, modular pipeline.

---

## 3. Dataset: CICIDS-2017

**Source**: Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset 2017 (CICIDS-2017), distributed via [Kaggle – dhoogla/cicids2017](https://www.kaggle.com/datasets/dhoogla/cicids2017).

### 3.1 Overview

CICIDS-2017 was generated over five days (Monday–Friday) in a controlled network environment simulating realistic enterprise traffic and a diverse set of modern attacks. The dataset consists of bidirectional network flow records extracted using CICFlowMeter, capturing 80+ statistical features per flow. Each record is labeled with the traffic type.

### 3.2 Traffic Categories

| Parquet File | Traffic Type | Day |
|---|---|---|
| `Benign-Monday-no-metadata.parquet` | Benign (normal) | Monday |
| `Bruteforce-Tuesday-no-metadata.parquet` | FTP/SSH Brute Force | Tuesday |
| `DoS-Wednesday-no-metadata.parquet` | DoS / Slowloris / Hulk / GoldenEye | Wednesday |
| `Infiltration-Thursday-no-metadata.parquet` | Infiltration + Web Attacks | Thursday |
| `WebAttacks-Thursday-no-metadata.parquet` | XSS / SQL Injection / Brute Force | Thursday |
| `Botnet-Friday-no-metadata.parquet` | ARES Botnet | Friday |
| `DDoS-Friday-no-metadata.parquet` | DDoS via LOIT | Friday |
| `Portscan-Friday-no-metadata.parquet` | Port Scan | Friday |

### 3.3 Class Imbalance

The combined dataset is severely imbalanced. Benign traffic constitutes the vast majority (~80%) of total flows. Attack categories like Infiltration contain only a few hundred samples, making standard accuracy metrics misleading and demanding specialized handling (oversampling, class-weighted losses, macro-F1 tracking).

### 3.4 Feature Space

Each flow record contains approximately 80 statistical features, including:

- **Flow-level aggregates**: Flow Duration, Total Forward/Backward Packets, Total Bytes
- **Packet length statistics**: Mean, Std, Min, Max for forward and backward packets
- **Inter-arrival times (IAT)**: Mean, Std, Min, Max for flow, forward, and backward directions
- **Flag counts**: SYN, FIN, RST, PSH, ACK, URG, CWE, ECE counts
- **Active/Idle time statistics**: Mean, Std, Min, Max
- **Subflow and bulk rate features**
- **Protocol field**: TCP (6), UDP (17), ICMP (1), Other

---

## 4. Data Preprocessing Pipeline

`01_cicids2017_data_processing.ipynb` implements a rigorous, multi-stage preprocessing pipeline that transforms raw parquet files into clean, normalized, feature-selected datasets ready for model training.

### 4.1 Hardware-Adaptive Environment Setup

The environment detection block configures TensorFlow for TPU (bfloat16 precision), multi-GPU (MirroredStrategy, float16), or CPU, with XLA JIT compilation enabled and reproducibility seeded at `SEED=42` across NumPy, Python random, and TensorFlow.

### 4.2 Data Loading and Integration

Eight parquet files are loaded independently using PyArrow and assigned a `source` label. They are concatenated into a single unified DataFrame for joint analysis and preprocessing, ensuring consistent column treatment across all traffic types.

### 4.3 Exploratory Data Analysis

- **Shape inspection**: row/column counts per sub-dataset
- **Missing value audit**: per-column and overall missing percentage visualized as bar charts
- **Basic statistics**: `describe()` for numerical columns and value counts for categorical fields
- **Class distribution**: combined label proportions displayed as a percentage bar chart, exposing the imbalance severity

### 4.4 Type Consistency Enforcement

A pattern-based enforcer scans column names for substrings like `Count`, `Packet`, `Bytes`, `Duration`, `Rate`, `Mean`, `Std`, `Max`, `Min` and coerces matching columns to numeric types via `pd.to_numeric(errors='coerce')`, converting impossible string values to NaN.

### 4.5 Protocol Encoding

The `Protocol` column (integer-coded: 6=TCP, 17=UDP, 1=ICMP) is one-hot encoded into dummy columns (`proto_6`, `proto_17`, `proto_1`, etc.) to prevent the model from inferring a spurious numeric ordering that would misrepresent the categorical protocol type.

### 4.6 Outlier and Validity Cleaning

Before any feature engineering, invalid flow records are removed:

- Infinite values (`np.inf`, `-np.inf`) in essential columns are replaced with NaN
- Flows with non-positive duration or negative packet counts are nullified
- Rows with missing values in any essential field are dropped

### 4.7 Feature Engineering

Several derived features are constructed to enrich the signal for anomaly and classification models:

| Feature | Formula | Rationale |
|---|---|---|
| `Total_Packets` | Fwd + Bwd packet counts | Overall flow volume |
| `Total_Bytes` | Fwd + Bwd byte totals | Bandwidth consumption |
| `Packet_Rate` | Total Packets / (Duration / 1e6) | Distinguishes high-rate attacks like DDoS |
| `Bytes_per_Packet` | Total Bytes / Total Packets | Average payload size |
| `Idle_CV` | Idle Std / Idle Mean | Coefficient of variation for idle times |
| `Active_CV` | Active Std / Active Mean | Coefficient of variation for active times |
| `Burst_Ratio` | Fwd Packets/s / Flow Packets/s | Asymmetry indicator, useful for DoS detection |
| `Packet_Entropy` | H(Fwd Std, Bwd Std) in bits | Flow irregularity; high entropy can indicate evasion |

Temporal features (`Hour`, `DayOfWeek`, `Is_Weekend`, `Peak_Hour`) are extracted from timestamps when available.

Packet entropy is computed vectorially using `scipy.stats.entropy` over normalized forward/backward packet length distributions, with caching to disk to avoid recomputation.

### 4.8 Feature Selection

A two-stage selection is applied:

**Stage 1 — Variance Threshold**: `VarianceThreshold(threshold=0.001)` removes near-constant features. From the retained features, a correlation matrix is computed and features with pairwise correlation > 0.9 are pruned (keeping the lower-indexed feature), eliminating redundant information.

**Stage 2 — Random Forest Importance**: A 100-tree Random Forest is trained on the variance-filtered features. Feature importances are computed and visualized; the top 85% most important features are retained, discarding the bottom 15% that contribute minimal predictive signal.

**Stage 3 — Optional PCA**: If more than 50 features remain after selection, PCA is applied retaining 95% of explained variance, projecting the data to a compact principal component space.

### 4.9 Label Encoding and Rare Class Handling

A `LabelEncoder` maps string labels to integer codes. Rare attack classes (those below a frequency threshold in the training set) are identified from the training split only — preventing data leakage — and optionally merged into an aggregate `Rare` category. A final stratified 80/20 train/test split is produced.

### 4.10 Artifact Export

The pipeline exports a structured artifact set to `/kaggle/working/cicids2017_processed/`:

- `data/train_processed.parquet`, `data/test_processed.parquet` — processed feature matrices with encoded labels
- `data/train_labels_original.parquet`, `data/test_labels_original.parquet` — original string label columns
- `artifacts/scaler.joblib` — fitted `QuantileTransformer` or `StandardScaler`
- `artifacts/label_encoder.joblib` — fitted `LabelEncoder`
- `artifacts/variance_selector.joblib` — fitted `VarianceThreshold`
- `artifacts/preprocess_settings.json` — JSON manifest with label mapping, selected features, and configuration
- `manifest.json` — top-level metadata manifest

---

## 5. Model Architectures

### 5.1 Variational Autoencoder (Anomaly Detection)

`02_vae_anomaly_detection.ipynb`

#### Concept

The first detection model is a semi-supervised anomaly detector: a VAE trained exclusively on benign traffic. The underlying hypothesis is that a VAE fitted on normal flows will reconstruct benign traffic faithfully while producing large reconstruction errors on attack flows, since attack patterns lie outside the learned latent distribution.

#### Architecture

The VAE consists of a symmetric encoder–decoder pair with LeakyReLU activations and BatchNormalization at every layer:

```
Input (D features)
    ↓
Dense(256) → BN → LeakyReLU
    ↓
Dense(128) → BN → LeakyReLU
    ↓
Dense(64)  → BN → LeakyReLU
    ↓
Dense(16) → z_mean     Dense(16) → z_log_var
           ↘               ↙
            Sampling (reparameterization)
                   ↓
              z ∈ ℝ¹⁶
                   ↓
Dense(64)  → BN → LeakyReLU
    ↓
Dense(128) → BN → LeakyReLU
    ↓
Dense(256) → BN → LeakyReLU
    ↓
Dense(D, linear) → x̂ (reconstruction)
```

**Latent dimension**: 16

#### Loss Function

The total VAE loss is:

```
L = L_recon + β · L_KL
```

where the reconstruction loss is a **feature-weighted mean squared error**:

```
L_recon = mean_i( w_i · (x_i - x̂_i)² )
```

Feature weights `w_i` are initialized as inverse-variance weights from the benign training set, normalized to unit mean, clipped to `[0.25, 4.0]`, and further fine-tuned based on FP/FN diagnostics (e.g., `Packet_Entropy` downweighted to 0.50, `Fwd Act Data Packets` upweighted to 1.20).

The KL divergence term uses **KL annealing**: weight is linearly ramped from 0 to 1 over 30 epochs via a `KLCallback`, preventing posterior collapse in early training.

#### Training Details

- Optimizer: Adam (`lr=1e-5`)
- Batch size: 2048 (benign flows only)
- Max epochs: 150 with early stopping (`patience=40`, `min_delta=0.01`, monitoring `val_loss`)
- 80/20 benign train/val split
- Stochastic sampling during training, deterministic (`z = z_mean`) at inference

#### Anomaly Scoring & Threshold

At inference, a **log-weighted reconstruction score** is computed:

```
score(x) = log1p( Σ_i w_i · (x_i - x̂_i)² )
```

The decision threshold is set at the **99th percentile** of benign validation scores. Flows exceeding the threshold are classified as attacks (binary: benign=0, attack=1). Binary ground truth is derived by collapsing all non-benign labels to 1.

---

### 5.2 VAE + LightGBM Gated System

`03_vae_lgbm_gated_classifier.ipynb`

#### Concept

The binary VAE detector is insufficient for operational deployment — security teams require precise attack category identification. This notebook introduces a **two-stage gated architecture** (`VAEGatedMultiClassSystem`) that separates anomaly detection from attack classification:

1. **Stage 1 (Gate)**: The pre-trained VAE computes a reconstruction error for each flow. Flows with error below the acceptance threshold (99th percentile of benign calibration errors, using raw MSE) are labeled benign and exit immediately.
2. **Stage 2 (Booster)**: Flagged flows are forwarded to a **LightGBM multi-class classifier**. The reconstruction error score is appended as an additional feature (augmented feature vector), giving the booster access to the VAE's uncertainty signal.

#### LightGBM Configuration

```python
LGBMClassifier(
    objective       = "multiclass",
    n_estimators    = 5000,
    learning_rate   = 0.03,
    num_leaves      = 63,
    min_child_samples = 40,
    subsample       = 0.85,
    colsample_bytree = 0.85,
    reg_lambda      = 1.0,
    n_jobs          = -1,
)
```

Class imbalance is handled via `compute_sample_weight(class_weight="balanced")` applied to both training and validation sets. Early stopping with `patience=100` evaluations is used.

#### Prediction Flow

```
Input Flow x
    ↓
VAE → recon_error
    ↓
error ≤ threshold? ──YES──→ Predict: BENIGN
    ↓ NO
Augment: [x_scaled ‖ recon_error]
    ↓
LightGBM → Multi-class prediction
```

This architecture provides an interpretable detection rationale: the VAE serves as a learned normality model while LightGBM excels at discriminative boundary learning for labeled classes.

---

### 5.3 Hierarchical VAE–LightGBM Gated System

`04_hierarchical_vae_lgbm_gated_ids.ipynb`

#### Concept

The `HierarchicalVAEGatedSystem` extends the two-stage architecture into a full ensemble where **a dedicated VAE is trained for every traffic class** (including each attack type). The system then uses a **learned meta-gate** to dynamically route predictions between LightGBM and the VAE ensemble.

#### Per-Class VAE Training

For each class `c` in the training set:
- If fewer than 50 samples exist, the class VAE is skipped (set to `None`)
- Classes in the `open_set_class_ids` set (default: class 7, typically Infiltration) are treated as open-set and skipped to avoid overfitting a VAE to out-of-distribution samples
- Minority classes with fewer than 5,000 samples are augmented via **noise-based oversampling**: existing samples are randomly selected and Gaussian noise (`σ=0.01`) is added to synthetic copies
- Each VAE shares the same architecture as in Section 5.1 (16-dim latent space, 256→128→64 encoder, 64→128→256 decoder)

#### Reconstruction Error Matrix

For an input batch of `n` samples, the system computes an `(n, C)` error matrix where entry `(i, c)` is the per-sample MSE between flow `i` and the reconstruction from class-`c`'s VAE. This matrix forms a rich discriminative feature set: the minimum error column indicates the most likely class, and the ratio between first and second minimum errors provides a confidence proxy.

#### Meta-Gate Features

A compact gate feature vector is derived from the error matrix:

- `min_err`: minimum reconstruction error across all class VAEs
- `second_min_err`: second minimum error
- `gap = second_min_err - min_err`: separation margin
- `ratio = second_min_err / (min_err + ε)`: relative confidence

These features feed a **3-way LightGBM meta-gate** trained to classify each sample into: (0) trust LightGBM, (1) trust VAE reconstruction argmin, or (2) abstain.

#### Main Booster

The primary **LightGBM multi-class booster** is trained on all labeled training data, optionally augmented with VAE latent representations if `use_latents=True`. When `use_latents=False` (default for speed), only the original feature space is used.

---

### 5.4 CatBoost Teacher + MLP Student Distillation

`05_catboost_mlp_knowledge_distillation.ipynb`

#### Concept

The final notebook implements **knowledge distillation** to compress the expressive power of a gradient-boosted tree ensemble (CatBoost) into a lightweight neural network (shallow MLP), enabling low-latency inference while preserving accuracy through a confidence-based routing fallback.

#### Teacher: CatBoostClassifier

```python
CatBoostClassifier(
    loss_function    = "MultiClass",
    iterations       = 800,
    learning_rate    = 0.03,
    depth            = 8,
    l2_leaf_reg      = 3.0,
    border_count     = 128,
    od_type          = "Iter",
    od_wait          = 50,
    use_best_model   = True,
)
```

The teacher is wrapped in `CatBoostTeacherWrapper` providing a unified `predict_proba(X)` interface and `save(path)` serialization. Class weights are provided to address imbalance.

#### Student: ShallowMLP

```python
Input(D) → Linear(128) → BN → ReLU → Dropout(0.15)
         → Linear(64)  → BN → ReLU → Dropout(0.15)
         → Linear(n_classes)
```

The student is implemented in **PyTorch** with mixed-precision training (`torch.cuda.amp.GradScaler`), AdamW optimizer (`lr=1e-3`, `weight_decay=3e-4`), and gradient clipping (`max_norm=1.0`).

#### Distillation Loss

The student is trained with a combined hard + soft loss:

```
L_total = α · L_hard + (1-α) · L_soft

L_hard = CrossEntropy(student_logits, ground_truth_labels)

L_soft = T² · KL( softmax(student_logits / T) ‖ teacher_probs )
```

where `α=0.5` and temperature `T=4.0`. Teacher probabilities are pre-computed once before training to avoid redundant forward passes. The KL divergence scales by `T²` following the Hinton et al. (2015) formulation, preserving the relative magnitude of soft targets.

#### Temperature Scaling (Calibration)

After distillation, the student's output distribution is calibrated using **temperature scaling**: a single scalar parameter `τ` is optimized via L-BFGS to minimize cross-entropy on the validation set:

```
calibrated_logits = student_logits / τ
```

This corrects for overconfident or underconfident probability estimates without modifying the model weights.

#### Confidence-Based Routing

A threshold `τ*` is tuned on the validation set via grid search over 60 evenly spaced values between the 1st and 99th percentiles of student confidence scores:

- If `max(student_probs) ≥ τ*`: use student prediction (fast path)
- If `max(student_probs) < τ*`: fall back to teacher prediction (accuracy path)

The grid search maximizes accuracy on the validation set while monitoring teacher usage rate, enabling a controllable accuracy–latency tradeoff. Best threshold and temperature are serialized to disk.

---

## 6. Training Strategy & Hyperparameters

| Component | Key Hyperparameters |
|---|---|
| VAE (all variants) | latent_dim=16, β=0.1, lr=1e-5, batch=2048, epochs≤150, KL annealing over 30 epochs |
| Feature-weighted Loss | inverse-variance init, clip [0.25, 4.0], hand-tuned factors per feature |
| Anomaly Threshold | 99th percentile of benign validation reconstruction scores |
| LightGBM (gate booster) | n_estimators=5000, lr=0.03, num_leaves=63, subsample=0.85, early stopping patience=100 |
| Per-class VAE ensemble | augment minority to 5000 samples, open-set class skipped, min_samples=20 |
| CatBoost (teacher) | iterations=800, lr=0.03, depth=8, l2_leaf_reg=3.0, od_wait=50 |
| MLP (student) | hidden=(128,64), dropout=0.15, AdamW lr=1e-3, wd=3e-4, T=4.0, α=0.5, patience=5 |
| Distillation temperature | T=4.0 during training; calibrated τ via L-BFGS post-training |
| Routing grid | 60 threshold candidates between conf[1%] and conf[99%] |

All experiments use `SEED=42` for NumPy, Python random, TensorFlow, and PyTorch.

---

## 7. Evaluation & Results Discussion

### 7.1 Metrics

Given severe class imbalance, primary metrics are:

- **Macro-F1**: unweighted average of per-class F1 scores; sensitive to rare class performance
- **Weighted-F1 / Accuracy**: overall correctness, dominated by benign majority
- **Per-class Precision, Recall, F1**: full classification report for each attack type
- **Confusion matrix**: visual inspection of misclassification patterns
- **ROC-AUC** (binary stage): area under the ROC curve for benign vs. attack
- **Teacher usage rate** (distillation stage): fraction of samples routed to the expensive teacher

### 7.2 VAE Anomaly Detector (Stage 1 Baseline)

The benign-trained VAE establishes a lower bound on binary detection capability. Its performance depends critically on threshold calibration: a 99th percentile threshold on benign validation scores controls the false positive rate at ~1%, but recall on sparse attack classes (Infiltration, Web Attacks) may be limited because some attack flows happen to produce low reconstruction errors — particularly protocol-conforming attacks. The feature-weighted loss and KL annealing improve the separation between benign and malicious reconstruction distributions compared to a vanilla MSE VAE.

### 7.3 VAE–LightGBM Gated System (Stage 2)

Augmenting LightGBM's input with the VAE reconstruction error adds a semantically rich uncertainty signal. The gated design substantially improves precision for benign traffic (routed at the VAE gate) while allowing LightGBM's discriminative power to handle the harder multi-class problem. Balanced sample weighting addresses class imbalance in the booster, improving recall on minority attack types.

The two-stage design reduces the effective classification problem size for LightGBM: it only processes flows flagged by the VAE gate, reducing computational load and focusing its capacity on ambiguous samples.

### 7.4 Hierarchical VAE Ensemble (Stage 3)

The per-class VAE approach reframes multi-class detection as competitive reconstruction: the class whose VAE best reconstructs a given flow is the predicted class. This is complementary to discriminative classification because it captures the generative manifold of each traffic type.

The meta-gate learns when the VAE ensemble is more reliable than LightGBM (high gap/ratio in reconstruction errors) versus when LightGBM's discriminative boundary is preferable (ambiguous reconstruction). This dynamic routing enables graceful degradation on out-of-distribution samples and provides a mechanism for open-set detection (class 7 / Infiltration is excluded from VAE training, forcing abstention for those samples).

### 7.5 Knowledge Distillation (Stage 4)

The CatBoost teacher's gradient-boosted representation is highly accurate but computationally expensive at inference (800 trees, depth 8). The distilled MLP student achieves competitive accuracy with a fraction of the inference cost. Key observations:

- The KL soft-loss compels the student to mimic inter-class probability distributions, not just hard labels — preserving nuanced decision boundary information
- Temperature scaling post-training corrects for student overconfidence, improving calibration and enabling reliable confidence-based routing
- The routing threshold produces a tunable Pareto tradeoff: lower thresholds route more to the teacher (higher accuracy, higher latency); higher thresholds rely more on the student (faster, with possible accuracy loss on ambiguous cases)

---

## 8. Dependencies & Environment

### Python & Core Libraries

| Library | Version (Tested) | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| TensorFlow | 2.13+ | VAE training (notebooks 2–4) |
| PyTorch | 2.0+ | MLP student (notebook 5) |
| LightGBM | 4.x | Multi-class booster (notebooks 3–4) |
| CatBoost | 1.2+ | Teacher model (notebook 5) |
| scikit-learn | 1.3+ | Preprocessing, metrics, selectors |
| pandas | 2.x | Data manipulation |
| numpy | 1.24+ | Numerical operations |
| scipy | 1.11+ | Entropy computation |
| pyarrow | 12+ | Parquet I/O |
| joblib | 1.3+ | Artifact serialization |
| matplotlib | 3.7+ | Plotting |
| seaborn | 0.13+ | Statistical visualization |
| scikit-image | 0.21+ | PSNR/SSIM metrics (optional) |
| umap-learn | 0.5+ | UMAP embedding (optional) |
| tensorflow-probability | 0.21+ | Probabilistic layers (notebook 2) |
| kagglehub | latest | Dataset download |

### Hardware

The environment setup auto-detects and configures:

- **TPU** (bfloat16, TPUStrategy) — primary target for VAE training
- **Multi-GPU** (float16, MirroredStrategy)
- **CPU** (float32 fallback)

XLA JIT compilation is enabled globally via `tf.config.optimizer.set_jit(True)`.

---

## 9. Reproduction Guide

### Step 1: Environment

```bash
pip install tensorflow lightgbm catboost scikit-learn pandas numpy scipy \
            pyarrow joblib matplotlib seaborn scikit-image umap-learn \
            tensorflow-probability torch torchvision kagglehub
```

### Step 2: Dataset

The notebooks use `kagglehub` to download CICIDS-2017 automatically. Alternatively, download manually from [Kaggle](https://www.kaggle.com/datasets/dhoogla/cicids2017) and update the `BASE_PATH` variable.

### Step 3: Run in Order

Execute the notebooks sequentially. Each notebook reads artifacts exported by its predecessor:

```
cicids2017_data_processing.ipynb
        ↓  exports: train/test parquets, scaler, label_encoder, settings.json
vae_anomaly_detection.ipynb
        ↓  exports: vae_cicids.keras, vae_threshold.joblib, vae_scaler.joblib
vae_lgbm_gated_classifier.ipynb
        ↓  exports: vae_lgbm_system.joblib
hierarchical_vae_lgbm_gated_ids.ipynb
        ↓  exports: hierarchical_system.joblib, per-class VAE weights
catboost_mlp_knowledge_distillation.ipynb
        ↓  exports: teacher_catboost.cbm, student_mlp_best.pt,
                    student_temperature.joblib, routing_threshold.joblib
```

All path variables (`BASE_PATH`, `MODEL_PATH`, etc.) are defined at the top of each notebook and must be updated if running outside of Kaggle.

---

## 10. Limitations & Future Work

**Current Limitations:**

- The VAE anomaly threshold is set heuristically (99th percentile). A learned threshold using a held-out calibration set would be more principled.
- The per-class VAE in notebook 4 treats open-set classes (Infiltration) by abstention, which may not generalize to truly unseen attack types at deployment.
- KL annealing schedule and feature weight tuning are partly manual and dataset-specific.
- Evaluation is confined to CICIDS-2017; cross-dataset validation (e.g., UNSW-NB15, CIC-IDS2018) is required to assess generalization.

**Future Directions:**

- **Online learning**: adapt the VAE threshold and LightGBM booster incrementally as new traffic patterns emerge without full retraining.
- **Attention-based encoder**: replace the MLP encoder with a transformer to capture long-range feature correlations in high-dimensional flow records.
- **Graph-based detection**: model network flows as nodes in a communication graph and apply GNN-based anomaly detection to capture lateral movement patterns.
- **Federated IDS**: train models across distributed network segments without centralizing raw traffic, preserving data privacy.
- **SHAP explainability**: add SHAP value analysis to provide human-interpretable feature attributions for each detection decision.

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{ids_cicids2017_2024,
  title     = {Deep Learning–Based Network Intrusion Detection System on CICIDS-2017},
  author    = {Yacer Meftah},
  year      = {2026},
  note      = {gh repo clone yacerak/Deep-Learning-Based-Network-Intrusion-Detection-System-on-CICIDS2017},
  url       = {https://github.com/yacerak/Deep-Learning-Based-Network-Intrusion-Detection-System-on-CICIDS2017.git}
}
```

**Dataset Reference**:

> Sharafaldin, I., Habibi Lashkari, A., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP)*, 108–116.

---

## License

This project is released under the GNU General Public License v3.0 . See `LICENSE` for details.
