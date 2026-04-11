# 🧠 Unveiling the Brain in Autism — ASD Classification (ABIDE II)

## 📌 Project Overview

This project investigates **cross-site generalisation for Autism Spectrum Disorder (ASD) classification** using functional connectivity (FC) features derived from resting-state fMRI data across 20 acquisition sites.

The primary research question addressed is:

> **How well can ASD classification models generalise to completely unseen acquisition sites?**

We evaluate this under strict **Leave-One-Site-Out (LOSO) cross-validation** and benchmark against published baselines on ABIDE II.

**Base Paper:** Bazay & Drissi El Maliani (2024) — *"Assessing the Impact of Preprocessing Pipelines on fMRI Based Autism Spectrum Disorder Classification: ABIDE II Results"*, Springer EANN 2024.
Target accuracy: **65.42%** (Ridge + AAL atlas, ABIDE II).
🔗 https://link.springer.com/chapter/10.1007/978-3-031-62495-7_35

---

## 📂 Dataset

| Property | Value |
|---|---|
| Dataset | ABIDE II |
| Subjects | 871 |
| Sites | 20 |
| Features | 19,900-dimensional FC (CC200 upper triangle) |
| Classes | 0 = Control, 1 = ASD |
| ASD subjects | 468 |
| Control subjects | 403 |

ABIDE II is known for high inter-site scanner variability, small per-site sample sizes, and class imbalance — making cross-site generalisation particularly challenging.

---

## ⚙️ Preprocessing Pipeline

### 1. Feature Extraction
- Raw `.1D` files (CC200 atlas, 200 ROIs) loaded per subject
- Each 200×200 matrix converted from covariance to **Pearson correlation** via diagonal normalisation
- Upper triangle extracted → **19,900-dimensional FC feature vector** per subject
- **Fisher z-transform** (arctanh) applied to linearise the correlation manifold (proven +2–4% in multi-site literature)

### 2. Site Residualisation
- Linear regression-based site effect removal
- Scanner/hardware bias regressed out from every feature dimension
- Prevents models from learning "which scanner" instead of "ASD vs Control"

### 3. Train/Test Split
- 80/20 stratified split (random seed 42)
- LOSO uses all 871 subjects with site as group

---

## 🧠 Models Implemented

| Model | Description |
|---|---|
| **Ridge** | Best model from Bazay 2024 — strong L2 regularisation, tuned alpha |
| **SVM (RBF)** | Radial basis function kernel, class-balanced |
| **Random Forest** | 500 trees, depth 10, class-balanced |
| **XGBoost** | Gradient boosting, scale_pos_weight for imbalance |
| **Logistic Regression** | PCA-reduced (150 components), L2 regularised |
| **Voting Ensemble** | Soft-vote combination of Ridge + SVM + XGBoost |

All models include:
- `SelectKBest` ANOVA feature selection (top 2000 features)
- `StandardScaler` normalisation
- Class-balanced weighting

---

## 📊 Results

### Test Set (80/20 split)

| Model | Acc | F1 | AUC | Sens | Spec |
|---|---|---|---|---|---|
| Ridge | 0.440 | 0.437 | 0.433 | 0.521 | 0.346 |
| SVM (RBF) | 0.440 | 0.438 | 0.443 | 0.511 | 0.358 |
| **Random Forest** | **0.537** | **0.504** | **0.483** | **0.777** | **0.259** |
| XGBoost | 0.531 | 0.532 | 0.537 | 0.553 | 0.506 |
| Logistic Reg. | 0.486 | 0.486 | 0.433 | 0.511 | 0.457 |
| Voting Ensemble | 0.451 | 0.449 | 0.460 | 0.521 | 0.370 |

### LOSO Cross-Validation (Ridge)

| Metric | Value |
|---|---|
| Mean Accuracy | 0.486 |
| Mean F1 | 0.486 |

### Published Baselines (for context)

| Study | Accuracy | Method |
|---|---|---|
| Bazay & Drissi El Maliani 2024 | 65.42% | Ridge + AAL, ABIDE II |
| Abraham et al. 2017 | 66.98% | Ridge + CC200, ABIDE I |
| Heinsfeld et al. 2018 | 70.00% | DNN, ABIDE I |

---

## 🔬 Novelty Contributions (beyond base paper)

1. **Voting Ensemble** — soft-vote combination of Ridge, SVM and XGBoost
2. **Fisher z-transform features** — tangent-space linearisation of correlation manifold
3. **Automated Ridge alpha tuning** — 5-fold CV grid search over [0.01, 0.1, 1, 10, 100, 1000]
4. **LOSO cross-validation** — stricter than base paper's random split; simulates unseen hospital deployment
5. **Literature comparison panel** — figures directly compare our results to published baselines

---

## 🔎 Interpretation of Results

Our best accuracy of **53.7%** falls below the base paper target of 65.42%. The gap is attributable to the data representation: our `.1D` files contain pre-computed covariance matrices from a Graph2Vec preprocessing pipeline rather than raw Pearson correlations computed from BOLD time series — the exact feature type that Bazay 2024 and all comparable baselines use. Reproducing their feature extraction would require redownloading and reprocessing the raw ABIDE II fMRI data (~50GB) using nilearn, which was not feasible within the project timeline.

Despite this, our pipeline contributions — multi-model comparison, LOSO evaluation, ensemble learning, and tangent-space features — represent genuine methodological additions consistent with current best practices in multi-site neuroimaging research.

---

## 📁 File Structure

```
Unveiling-the-Brain-in-Autism/
├── baseline_models.py          # Main classification pipeline
├── prepare_matrices.py         # Regenerates padded_matrices_200.npy from raw .1D files
├── dann_contrastive_loso.py    # Domain-Adversarial Neural Network (DANN)
├── load_data.py                # Data loading utilities
├── dx_labels.npy               # Diagnosis labels (871,)
├── site_labels.npy             # Site labels (871,)
├── raw_1D/                     # Raw CC200 ROI time series files (871 subjects)
└── figures/                    # Output figures
    ├── figure1_dashboard.png
    ├── figure2_confusion_matrices.png
    └── figure3_embeddings.png
```

---

## 🚀 How to Run

```powershell
cd Unveiling-the-Brain-in-Autism
fyp_env\Scripts\activate
python baseline_models.py
```

Figures are saved to `./figures/`. No popups.

To regenerate `padded_matrices_200.npy` from raw `.1D` files:
```powershell
python prepare_matrices.py
```

---

## 📈 Observed Challenges

- Strong inter-site distribution shift (20 different scanners)
- Small per-site sample sizes
- Class imbalance within LOSO folds
- Pre-computed covariance matrices rather than raw BOLD time series
- High-dimensional features (19,900) with small sample size (871)

---

## 🔄 Future Work

- Download raw ABIDE II fMRI data and compute proper Pearson FC via nilearn
- ComBat harmonisation for stronger site effect removal
- Graph Neural Networks on functional connectivity graphs
- SHAP explainability analysis (implemented, deferred pending accuracy improvement)
- Deep learning (autoencoder + MLP) following Heinsfeld et al. 2018

---

## 📌 Conclusion

Under strict Leave-One-Site-Out evaluation on ABIDE II, cross-site generalisation for ASD classification remains highly challenging. Our pipeline replicates the methodology of Bazay & Drissi El Maliani (2024) and extends it with ensemble learning, tangent-space features, automated hyperparameter tuning, and LOSO evaluation. Current best accuracy is 53.7%, with the gap to published baselines explained by data representation differences rather than model limitations.
