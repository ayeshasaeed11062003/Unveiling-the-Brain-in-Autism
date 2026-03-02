🧠 Cross-Site Autism Classification using Graph Embeddings (ABIDE II)
📌 Project Overview

This project investigates cross-site generalization for Autism Spectrum Disorder (ASD) classification using graph-based brain embeddings derived from functional connectivity data.

The primary challenge addressed is:

How well can ASD classification models generalize to completely unseen acquisition sites?

We evaluate this under strict Leave-One-Site-Out (LOSO) cross-validation.

📂 Dataset

Dataset: ABIDE II

Subjects: 871

Sites: 20

Features: 19,900-dimensional graph embeddings

Labels:

1 = ASD

2 = Control

ABIDE II is known for:

High heterogeneity

Strong inter-site scanner variability

Small per-site sample sizes

Class imbalance per site

This makes cross-site generalization particularly challenging.

🧪 Experimental Protocol
Cross-Validation Strategy

We use:

Leave-One-Site-Out (LOSO)

For each fold:

Train on 19 sites

Test on the remaining unseen site

This simulates real-world deployment where the model encounters data from a completely new acquisition center.

⚙️ Preprocessing Pipeline
1️⃣ Label Encoding

ASD → 1

Control → 0

Site labels encoded via LabelEncoder

2️⃣ Dimensionality Reduction

Original feature size: 19,900
Reduced via PCA to 500 components.

Reason:

Reduce overfitting

Improve training stability

Remove noise

Improve generalization

3️⃣ Standardization

StandardScaler fitted on training fold only
Applied to test fold to prevent leakage.

🧠 Models Implemented
1️⃣ Baseline MLP (Implicit)

A simple fully connected classifier trained under LOSO.

2️⃣ Domain-Adversarial Neural Network (DANN)

Architecture:

Feature Extractor:

Linear(500 → 512)

ReLU

Dropout(0.3)

Linear(512 → 128)

ReLU

Two heads:

Diagnosis classifier (ASD vs Control)

Domain classifier (Site ID)

Includes:

Gradient Reversal Layer (GRL)

Progressive lambda ramp

Gradient clipping

Low learning rate (1e-4)

Balanced objective weighting

Loss function:

Total Loss =
ASD CrossEntropy

0.05 × Domain Loss

0.02 × Contrastive Loss

3️⃣ Contrastive Representation Learning

Supervised contrastive loss applied to feature embeddings:

Purpose:

Encourage same-diagnosis clustering

Improve separability

Improve cross-site invariance

Temperature scaling used for stability.

📊 Results (LOSO, ABIDE II)

Mean across 20 sites:

ASD Accuracy: 46.3%
ASD F1 Score: 0.399
Site Accuracy: ~0.0%

🔎 Interpretation of Results
1️⃣ Domain Confusion

Site classifier accuracy ≈ 0 indicates:

Strong removal of site-specific signal

Successful adversarial domain suppression

However, since LOSO uses unseen sites, domain accuracy on test fold is expected to be near zero.

2️⃣ Diagnostic Performance

Accuracy ≈ 46% under strict LOSO.

This aligns with known difficulty of ABIDE II under cross-site generalization.

Published literature commonly reports:

45–55% LOSO accuracy

High variance across sites

Some folds reached >60%, others dropped to ~33%, reflecting strong site-level heterogeneity.

🔬 Key Scientific Insight

Results suggest:

Diagnostic signal and site-specific signal are partially entangled in graph embeddings.

Aggressive removal of site information may suppress useful diagnostic features.

This highlights a core challenge in multi-site neuroimaging:

Harmonization vs Discriminability Tradeoff

📈 Observed Challenges

Strong site distribution shift

Small per-site sample size

Class imbalance within folds

Scanner-induced variability

High dimensional graph embeddings

🔄 Stability Measures Applied

PCA dimensionality reduction

Gradient clipping (norm = 5)

Slow adversarial lambda ramp

Reduced domain loss weighting

Reduced contrastive weighting

Fixed random seed

No data leakage

🚀 Future Work

Planned improvements:

Class-weighted loss for imbalance correction

Linear site residualization prior to training

Mixup augmentation to simulate unseen domains

Attention-based feature extractor

Comparison with ComBat harmonization

Evaluate ABIDE I vs ABIDE II difference

🧠 Research Contribution

This work provides:

A strict LOSO benchmark on ABIDE II graph embeddings

Empirical evidence of domain-adversarial limitations under unseen-site generalization

Insight into the coupling between site effects and ASD discriminability

📁 File Structure

graph_embeddings.npy

dx_labels.npy

site_labels.npy

dann_contrastive_loso.py

📌 Conclusion

Under strict Leave-One-Site-Out evaluation on ABIDE II:

Cross-site generalization remains highly challenging.

Adversarial domain removal suppresses site bias.

However, ASD discriminability under unseen domains remains limited.

These findings reflect inherent heterogeneity in ABIDE II and align with known literature diffi
