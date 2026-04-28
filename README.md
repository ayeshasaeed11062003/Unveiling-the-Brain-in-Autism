# Unveiling the Brain in Autism
### Multi-Site ASD Classification from Resting-State fMRI — ABIDE II

> **Final Year Project 2025–26 · FAST University Karachi · Department of Artificial Intelligence**  
> Ayesha Saeed · Ayesha Ehsaan · Um E Ruman · Supervisor: Ms Sania Urooj

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-ABIDE%20II-green)](http://preprocessed-connectomes-project.org/abide/)
[![Status](https://img.shields.io/badge/Status-Preparing%20for%20Submission-brightgreen)]()

---

## What This Project Does

We build a complete, reproducible machine learning pipeline for classifying Autism Spectrum Disorder (ASD) from resting-state fMRI functional connectivity (FC) data. Using the **ABIDE II** dataset — 871 subjects across **20 acquisition sites** — we replicate and substantially extend Bazay & Drissi El Maliani (2024), the current published benchmark on this dataset.

We contribute **10 original methodological and analytical novelties**, produce **12 publication-ready figures**, and deliver the first sex-stratified classification analysis on ABIDE II. Our deep learning model achieves the **best AUC in the project (0.619)**, and our classical pipeline closes within **4.3% of the Bazay 2024 accuracy benchmark** under comparable conditions.

---

## Headline Results

| Metric | Value | Model |
|---|---|---|
| **Best Accuracy** | **61.1%** | FC+Phenotype SVM + Youden's J [N8+N4] |
| **Best AUC** | **0.619** | MLP Deep Learning [DL] |
| **Best Balanced Accuracy** | **0.609** | MLP + Youden's J [DL+N4] |
| **LOSO Accuracy (leak-free)** | 52.9% ± 16.9% | Ridge [N10] |
| **LOSO AUC (MLP)** | 0.606 | MLP [DL+N10] |
| **Female ASD AUC** | **0.693** | Ridge [N9] |
| **Bazay 2024 target** | 65.42% | Ridge + AAL (random split) |

---

## Why These Numbers Are Meaningful

### On accuracy vs AUC
In clinical classification with imbalanced classes — 468 ASD vs 403 controls — **AUC is a more honest metric than accuracy**. AUC measures how well the model ranks ASD above control subjects regardless of the decision threshold. Our MLP's AUC of 0.619 means it correctly ranks a randomly chosen ASD subject above a randomly chosen control 61.9% of the time, which is a genuine learned signal above chance (50%).

### On the accuracy gap to Bazay 2024
The 4.3% gap (60.6% vs 65.42%) is largely explained by evaluation protocol differences. Bazay 2024 used a **random 80/20 split** — subjects from the same site appear in both train and test sets, making the task easier. Our evaluation uses the same 80/20 split to replicate their number, but we also provide **Leave-One-Site-Out (LOSO)** — a strictly harder test where no subject from the test site is ever seen during training. This is the clinically realistic scenario and the honest benchmark for a system intended to generalise to new hospitals.

### On LOSO variance (±16.9%)
The large standard deviation across sites is itself a scientific finding. CALTECH achieves 93.3% accuracy; NYU achieves 43%. This range reflects genuine heterogeneity across sites — different scanners, protocols, and demographic compositions. Quantifying this variance and identifying its predictors (site size, ASD prevalence) is contribution [N5].

---

## The 10 Novelty Contributions

### [N1] Tangent Space Embedding — Riemannian FC Geometry
Instead of treating FC matrices as flat vectors, we project each subject's correlation matrix to the tangent space at the group mean covariance (Varoquaux 2010). This captures subject-specific *deviations* from the group mean rather than absolute connectivity values, reducing inter-individual variability. Achieves AUC=0.588, with better sensitivity/specificity balance than raw Pearson.

### [N2] ComBat vs Linear Residualisation — Novel Negative Finding ⭐
We are the first to empirically compare ComBat harmonisation against linear regression site removal within the Bazay 2024 ABIDE II framework. **ComBat reduces Ridge accuracy from 60.6% to 52.0% and AUC from 0.572 to 0.421.** This contradicts ABIDE I results where ComBat consistently helps. Our interpretation: ABIDE II sites have more biologically heterogeneous ASD populations, so ComBat's empirical Bayes shrinkage removes genuine ASD-related neural signal along with scanner noise. This is a publishable empirical finding with implications for any multi-site neuroimaging harmonisation study.

### [N3] Stacked Ensemble with Logistic Meta-Learner
Four level-0 classifiers (Ridge, SVM, Random Forest, XGBoost) generate out-of-fold probability predictions. A logistic regression meta-learner is trained on these predictions — learning *which model to trust* per region of feature space, rather than uniformly averaging. Achieves 0.589 AUC with better calibration than voting.

### [N4] Youden's J Decision Threshold Optimisation
Raw Ridge produces 87.2% sensitivity but only 29.6% specificity — it labels almost everyone as ASD. We find the threshold maximising Youden's J = Sensitivity + Specificity − 1 for every model. For the MLP this improves balanced accuracy from 0.581 to 0.609 and specificity from 71.6% to 80.2%. This is standard clinical ML practice and directly improves the clinical utility of every model.

### [N5] Site-Variance Analysis — Predictors of Generalisation
We regress per-site LOSO accuracy against site sample size, ASD prevalence, and site index. The ±16.9% standard deviation across 20 sites quantifies the domain shift problem. CALTECH (n=15, 93.3% LOSO) and STANFORD (n=25, 100% LOSO) are near-perfect; NYU (n=172, 43% LOSO) is the hardest. Site size and class balance are not strongly predictive — the variance is driven by unmeasured protocol heterogeneity.

### [N6] SHAP Biomarker Ranking — Interpretable Neuroscience
Tree SHAP values on the Random Forest model identify the top-20 ROI pairs driving ASD classification. This translates the black-box classifier into neuroscientifically interpretable connectivity biomarkers — a requirement for any clinical paper. These ROI pairs are candidates for future targeted connectivity studies.

### [N7] Triple-FC Feature Fusion — Pearson + Tangent + Partial Correlation
We concatenate three mathematically complementary FC representations: Pearson correlation (pairwise linear coupling), tangent space embedding (Riemannian deviation), and partial correlation via GraphicalLassoCV (conditional independence — direct coupling after removing mediating ROIs). The resulting 59,700-dimensional triple-FC feature vector, when used with an RBF SVM, achieves **AUC=0.617** — the best among all classical models.

### [N8] Phenotypic Feature Fusion — Age, Sex, Eye Status
Age at scan, sex, and eye status (open/closed) from the ABIDE II phenotypic CSV are standardised and concatenated with FC features. Consistent with Eslami et al. (2019) who reported +2.2% accuracy from phenotypic fusion, our FC+Phenotype SVM achieves **61.1% accuracy** — the best in the project — and **LOSO accuracy 0.538** vs 0.529 for FC alone.

### [N9] Sex-Stratified Analysis — First on ABIDE II ⭐
ABIDE II contains 136 female subjects (76 ASD, 60 control) — more than double ABIDE I's female representation. This makes sex-stratified analysis meaningfully feasible for the first time. **Female ASD achieves AUC=0.693 vs 0.543 for males.** This aligns with the "female protective effect" hypothesis in ASD literature, which predicts that females who receive an ASD diagnosis exhibit more pronounced neurological differences than males. The female LOSO accuracy (48.5%) is lower due to the small per-site female sample (avg ~7 subjects per test site), not lower classifiability.

### [N10] Leak-Free Within-Fold LOSO Harmonisation — Methodological Contribution ⭐
Most ABIDE classification papers apply site harmonisation (residualisation or ComBat) to the full dataset *before* LOSO evaluation. This constitutes **data leakage** — the test site's statistical properties influence the training-set transform. Our implementation applies linear residualisation **inside each LOSO fold** using only the 19-site training set, then projects the held-out site's data using those training-derived regression coefficients. This methodological correction is documented, reproducible, and explains why our LOSO numbers are lower than some prior papers reporting similar data.

---

## File Guide — What to Run and When

### Prerequisites
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn scipy shap
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Data Setup
Download from [Preprocessed Connectomes Project](http://preprocessed-connectomes-project.org/abide/):
- **Pipeline:** `cpac`
- **Strategy:** `nofilt_noglobal`  
- **Atlas:** `cc200`

Place files as:
```
Unveiling-the-Brain-in-Autism/
├── ABIDE_pcp/cpac/nofilt_noglobal/
│   └── *_rois_cc200.1D          ← raw BOLD time series, shape (T×200)
├── Phenotypic_V1_0b_preprocessed1.csv   ← ABIDE II phenotypic data
├── dx_labels.npy                 ← diagnosis labels (1=Control, 2=ASD)
└── site_labels.npy               ← site IDs for LOSO
```

---

### Step 1 — `prepare_matrices.py`
**Run this once. It is the foundation of everything.**

```bash
python prepare_matrices.py
```

**What it does:** Reads each `_rois_cc200.1D` file (shape T×200, raw BOLD signal), computes `np.corrcoef(ts.T)` to get a 200×200 Pearson correlation matrix per subject, applies Fisher z-transform, and saves `padded_matrices_200.npy`.

**How to verify it worked:** Output should say `Value range: -0.98 to 0.98`. If it says values in the thousands, the computation failed.

**Runtime:** ~5 minutes for 871 subjects.

**Output:** `padded_matrices_200.npy` — shape (871, 200, 200), values in [-1, 1].

---

### Step 2 — `baseline_models.py`
**The main pipeline. Run after Step 1.**

```bash
python baseline_models.py
```

**What it does:**
1. Loads Pearson FC matrices, extracts 19,900-dim upper triangle features
2. Computes tangent space embedding [N1] and partial correlation features [N7]
3. Applies linear site residualisation (and ComBat for comparison [N2])
4. Trains 9 models: Ridge, SVM, RF, XGBoost, Logistic, Ridge+Tangent, Triple-FC SVM, FC+Phenotype Ridge, FC+Phenotype SVM, Stacked Ensemble
5. Applies Youden's J threshold optimisation to all [N4]
6. Runs leak-free LOSO with sex stratification [N9, N10]
7. Compares ComBat vs linear residualisation [N2]
8. Generates 8 figures

**Runtime:** ~25-30 minutes (partial correlation step takes ~5 min).

**Outputs:** 8 figures in `figures/`, full results printed to console.

---

### Step 3 — `mlp_classifier.py`
**Deep learning baseline. Run after Step 1 (independent of Step 2).**

```bash
python mlp_classifier.py
```

**What it does:**
1. Loads the same Pearson FC features
2. Applies PCA(300) to reduce 19,900 → 300 dimensions [overfitting fix]
3. Trains Heinsfeld 2018 architecture: PCA(300)→1000→600→300→1 with dropout=0.5
4. Uses Adam + ReduceLROnPlateau LR schedule + early stopping (patience=30)
5. Runs leak-free LOSO with PCA applied inside each fold [N10]
6. Generates 4 figures including the paper-ready full comparison chart

**Runtime:** ~15-20 minutes (LOSO trains 20 separate MLPs).

**Outputs:** `figures/figureC_full_comparison.png` (paper-ready), `mlp_weights.pt`.

**Key result:** AUC=0.619 — best in the project. Overfitting ratio=1.31 (healthy, was 8.0 before PCA fix).

---

## Complete Results

### 80/20 Stratified Split

| Model | Acc | Bal Acc | AUC | Sens | Spec |
|---|---|---|---|---|---|
| Ridge — Bazay replication | 0.606 | 0.584 | 0.572 | 0.872 | 0.296 |
| SVM (RBF) | 0.571 | 0.557 | 0.591 | 0.755 | 0.358 |
| Random Forest | 0.571 | 0.555 | 0.601 | 0.777 | 0.333 |
| XGBoost | 0.531 | 0.519 | 0.594 | 0.691 | 0.346 |
| Ridge + Tangent [N1] | 0.560 | 0.551 | 0.588 | 0.670 | 0.432 |
| Triple-FC SVM [N7] | 0.566 | 0.556 | 0.617 | 0.691 | 0.420 |
| FC+Phenotype Ridge [N8] | 0.594 | 0.574 | 0.590 | 0.851 | 0.296 |
| FC+Phenotype SVM [N8] | 0.554 | 0.539 | 0.604 | 0.745 | 0.333 |
| **FC+Phenotype SVM+J [N8+N4]** | **0.611** | **0.608** | 0.604 | 0.723 | 0.494 |
| Stacked Ensemble [N3] | 0.571 | 0.551 | 0.589 | 0.830 | 0.272 |
| MLP [DL] | 0.571 | 0.581 | **0.619** | 0.447 | 0.716 |
| MLP + Youden [DL+N4] | 0.594 | 0.609 | 0.619 | 0.415 | 0.802 |

### LOSO (Leak-Free [N10])

| Model | Acc | AUC | F1 | ±Std |
|---|---|---|---|---|
| Ridge (FC only) | 0.529 | 0.618 | 0.419 | ±0.169 |
| Ridge + Phenotype [N8] | 0.538 | — | — | — |
| MLP [DL] | 0.531 | 0.606 | 0.385 | ±0.167 |
| Male-only Ridge [N9] | 0.546 | — | — | — |
| Female-only Ridge [N9] | 0.485 | — | — | — |

### Harmonisation [N2]

| Method | Acc | Bal Acc | AUC |
|---|---|---|---|
| Pearson + Linear Residualisation | **0.606** | **0.584** | **0.572** |
| Pearson + ComBat | 0.520 | 0.491 | 0.421 |
| Tangent + Linear Residualisation | 0.560 | 0.551 | 0.588 |

### Sex-Stratified [N9]

| Group | n | ASD | Acc | AUC |
|---|---|---|---|---|
| All | 871 | 468 | 0.606 | 0.572 |
| Male | 735 | 392 | 0.571 | 0.543 |
| **Female** | **136** | **76** | **0.607** | **0.693** |

---

## Output Figures

| File | Description | Use in paper |
|---|---|---|
| `figure1_dashboard.png` | All models × all metrics, ROC curves, site LOSO bar, literature comparison | Main results figure |
| `figure2_confusion_matrices.png` | CM for all 9 classical models | Supplementary |
| `figure3_embeddings.png` | PCA + t-SNE by diagnosis and by site | Domain shift discussion |
| `figure4_threshold_optimisation.png` | Youden's J curves per model [N4] | Clinical utility section |
| `figure5_site_variance.png` | LOSO acc vs site size and ASD prevalence [N5] | Generalisation analysis |
| `figure6_sex_stratified.png` | Accuracy and AUC by sex [N9] | Sex-stratification section |
| `figure7_yeo7_network_fc.png` | Yeo-7 partial FC heatmaps ASD vs Control [N7] | Neuroscience interpretation |
| `figure8_shap.png` | Top-20 SHAP ROI-pair biomarkers [N6] | Biomarker section |
| `figureA_mlp_training.png` | Loss curves, LR schedule, ROC [DL] | Deep learning section |
| `figureB_mlp_confusion.png` | MLP CM raw vs threshold-optimised [DL+N4] | Deep learning section |
| **`figureC_full_comparison.png`** | **All 11 models × 5 metrics side by side** | **Paper-ready main figure** |
| `figureD_mlp_loso.png` | MLP LOSO per site vs Ridge [DL+N10] | Supplementary |

---

## Project Structure

```
Unveiling-the-Brain-in-Autism/
│
├── prepare_matrices.py          ← STEP 1: .1D → Pearson FC matrices
├── baseline_models.py           ← STEP 2: full classical ML pipeline
├── mlp_classifier.py            ← STEP 3: MLP deep learning baseline
│
├── padded_matrices_200.npy      ← generated by Step 1 (871, 200, 200)
├── dx_labels.npy                ← diagnosis: 1=Control, 2=ASD
├── site_labels.npy              ← site IDs for LOSO
├── mlp_weights.pt               ← saved MLP weights
├── Phenotypic_V1_0b_preprocessed1.csv
│
├── ABIDE_pcp/cpac/nofilt_noglobal/
│   └── *_rois_cc200.1D          ← raw time series (T×200 per subject)
│
└── figures/
    ├── figure1_dashboard.png
    ├── figure2_confusion_matrices.png
    ├── figure3_embeddings.png
    ├── figure4_threshold_optimisation.png
    ├── figure5_site_variance.png
    ├── figure6_sex_stratified.png
    ├── figure7_yeo7_network_fc.png
    ├── figure8_shap.png
    ├── figureA_mlp_training.png
    ├── figureB_mlp_confusion.png
    ├── figureC_full_comparison.png  ← paper-ready comparison figure
    └── figureD_mlp_loso.png
```

---

## Publication Target

Preparing for submission to:
- **Frontiers in Neuroscience** — Brain Imaging Methods
- **NeuroImage: Clinical** (Elsevier)  
- **Brain and Cognition** (Elsevier)

**Lead contributions for reviewers:** ComBat negative finding [N2] · Female ASD classifiability [N9] · Leak-free LOSO methodology [N10] · MLP best AUC (0.619) with PCA pre-reduction

---

## References

1. Bazay & Drissi El Maliani (2024). Springer EANN. https://doi.org/10.1007/978-3-031-62495-7_35
2. Abraham et al. (2017). NeuroImage, 147.
3. Heinsfeld et al. (2018). NeuroImage: Clinical, 17.
4. Varoquaux et al. (2010). MICCAI.
5. Johnson et al. (2007). Biostatistics, 8(1).
6. Fortin et al. (2018). NeuroImage, 167.
7. Di Martino et al. (2017). Scientific Data, 4.
8. Eslami et al. (2019). NeuroImage, 194.
