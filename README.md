# Unveiling-the-Brain-in-Autism
An informed approach to Autism diagnosis

# 🧠 Radiomics-Based Contrastive Learning for Autism Diagnosis

**Final Year Project – FAST-NUCES, Karachi (2025)**

## 📄 Overview

This project proposes an explainable, multi-stage machine learning pipeline for **Autism Spectrum Disorder (ASD) diagnosis** and **phenotype prediction** using the **ABIDE PCP** (Autism Brain Imaging Data Exchange – Preprocessed Connectomes Project) dataset.

Each brain is represented as a **graph** derived from functional MRI (fMRI) connectivity matrices. Radiomics-style graph features — such as clustering, global efficiency, and small-worldness — are extracted, and **contrastive self-supervised learning** is applied to learn **site-invariant embeddings**. These embeddings are then fine-tuned for:

* **ASD vs. Control classification**, and
* **Phenotype prediction** (e.g., age, IQ).

Finally, **SHAP explainability** highlights critical brain subnetworks influencing predictions, bridging interpretability with diagnostic power.


## ⚙️ Pipeline Summary

| Stage                               | Description                                                           | Lead          |
| ----------------------------------- | --------------------------------------------------------------------- | ------------- |
| **1. Data Loading**                 | Fetch ABIDE fMRI connectomes via `nilearn.datasets.fetch_abide_pcp()` | Ayesha Ehsaan |
| **2. Graph Construction**           | Convert 200×200 connectivity matrices into graphs                     | Ayesha Ehsaan |
| **3. Radiomics Feature Extraction** | Extract clustering, efficiency, small-worldness, hubness metrics      | Ayesha Ehsaan |
| **4. Contrastive Learning**         | Learn robust embeddings using SimCLR-style self-supervised training   | Ayesha Saeed  |
| **5. Classification & Regression**  | Fine-tune on ASD vs. Control and predict phenotypes                   | Ayesha Saeed  |
| **6. Explainability**               | Apply SHAP to identify brain subnetworks influencing diagnosis        | Ruma          |

## 🧩 Technologies Used

* **Python 3.10+**
* **Libraries:**
  `nilearn`, `networkx`, `numpy`, `pandas`, `matplotlib`, `seaborn`,
  `torch`, `scikit-learn`, `shap`, `tqdm`

## 🧠 Key Contributions

✅ First integrated pipeline combining:

* Radiomics-inspired graph features
* Contrastive self-supervised embeddings
* Explainable AI (SHAP) for ASD diagnosis

✅ Tackles **multi-site scanner variability** (ABIDE’s biggest challenge)
✅ Moves toward **interpretable, clinically relevant biomarkers**

## 📊 Outputs

* Radiomics feature dataset (`features_df`)
* Learned embeddings from contrastive model
* ASD vs Control classifier results
* SHAP explainability plots
* Optional phenotype regression results (e.g., age prediction)

## 🚀 Next Steps

* Add domain adaptation for stronger site-invariance
* Integrate GNNExplainer for subnetwork-level interpretability
* Extend to multimodal MRI (structural + functional)
