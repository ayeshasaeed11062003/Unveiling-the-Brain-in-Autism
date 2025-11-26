# Unveiling the Brain in Autism  
### Graph-Based Analysis of Functional Connectivity in the ABIDE Dataset  
**FYP-1: Baseline Data Processing, Graph Embeddings & Site-Bias Analysis**  
**FYP-2: Contrastive Learning for Site-Generalizable Brain Representations (Upcoming)**

---

## Overview
This project analyzes functional connectivity (FC) matrices from the **ABIDE** (Autism Brain Imaging Data Exchange) dataset to investigate:

1. Whether FC-derived brain graphs contain meaningful **diagnostic signal** (ASD vs Control).  
2. Whether strong **site effects** (scanner/site differences) distort the learned embeddings.  
3. How to make these representations **generalizable across sites** using advanced contrastive learning techniques (planned for FYP-2).

FYP-1 focuses on building a solid preprocessing pipeline, generating graph embeddings, visualizing structure using t-SNE, and identifying the fundamental limitations (strong site bias and weak ASD signal) that motivate FYP-2.

---

## Dataset Used

### **ABIDE Preprocessed (CPAC / nofilt_noglobal)**
- 871 functional connectivity matrices  
- 200 × 200 ROI connectivity (CC200 atlas)  
- Subject-level metadata from phenotype CSV (`Phenotypic_V1_0b_preprocessed1.csv`)
- Diagnosis label:  
  - `1 = ASD`  
  - `2 = Control`  
- Site label: 20+ scanning sites across institutions (NYU, UCLA, TRINITY, PITT, etc.)

---

## Pipeline Summary

### **1. Load Padded Matrices**
All FC matrices are loaded from the `.npy` padded file created earlier:
- Shape: **871 × 200 × 200**
- Fully aligned in the same ROI order.

---

### **2. Filename → Subject ID Matching**
Each original `.1D` connectivity file contains a subject identifier (e.g. `CMU_a_0050642_rois_cc200.1D` → `0050642`).

A custom ID extractor was written to:
- Parse all files in the CPAC folder
- Extract numeric subject IDs
- Match to phenotype rows

Matched **871 / 871** files → perfect alignment.

---

### **3. Phenotype Alignment**
Loaded phenotype CSV:
- 1112 rows
- Converted all `SUB_ID` to strings
- Intersected with extracted file IDs

Found **871 valid subjects**  
**All 871 subjects** have a valid `DX_GROUP` (diagnosis)

---

### **4. Convert Matrices → Graphs**
Each 200×200 matrix is thresholded:
- `adj[i,j] = 1` if connectivity > 0.3  
- Converted to undirected NetworkX graph  
- Removed self-loops  

Built **871 graphs**

---

### **5. Graph Embeddings (Baseline)**
For each graph:
- Extracted **upper triangle** (~19,900 features per subject)
- Saved as `graph_embeddings.npy`

Shape:  
(871 subjects, 19900 features)


These embeddings serve as the baseline.

---

## Visualization & Findings (t-SNE)

Three t-SNE plots were generated to visualize embedding structure.

---

## **1. t-SNE Colored by Diagnosis (ASD vs Control)**

### **Finding:**  
ASD and Control subjects **do not form distinct clusters**.  
Both classes are fully mixed and overlapping.

### **Meaning:**  
The functional connectivity graphs **do not contain strong diagnosis signal**  
OR  
The current embedding method is not capturing ASD-relevant patterns.

This is consistent with ABIDE’s known difficulty: ASD patterns are subtle and noisy.

---

## **2. t-SNE Colored by Site (Scanner/Location)**

### **Finding:**  
Subjects cluster **very strongly by site**.  
Clusters form around specific institutions (e.g., UCLA, NYU, TRINITY).

### **Meaning:**  
**Site effects dominate the variance** in the embeddings.  
The scanner/site identity appears to be a much stronger signal than the ASD diagnosis.

This is exactly the expected ABIDE problem:
> “Models learn the site, not the autism.”

---

## **3. t-SNE for Top 10 Largest Sites**

### **Finding:**  
The structure remains:
- Tight clusters for each site
- Minimal overlap between sites
- No grouping by diagnosis

### **Meaning:**  
Even when visualizing only high-sample sites:
- **The embeddings are not diagnosis-informative**
- **Site bias remains extremely strong**

This confirms we absolutely need domain-generalization.

---

# Summary of FYP-1 Findings

### ✔ Data pipeline complete  
### ✔ Graph construction complete  
### ✔ Embeddings generated  
### ✔ t-SNE visualizations successfully produced  
### ✔ Diagnostic signal is weak  
### ✔ Site bias dominates the embedding space  
### ✔ Strong justification for contrastive learning in FYP-2

---

# **Conclusion of FYP-1**
FYP-1 successfully establishes the baseline pipeline and uncovers critical challenges:

1. **ASD cannot be separated from controls using simple graph embeddings.**  
2. **Site effects dominate**, meaning scanners confound all downstream ML models.  
3. **This dataset requires site-invariant representation learning**, not classical ML.

These results set the stage for a more advanced, contrastive-learning-based solution.

---

# FYP-2: Next Steps  
### **Goal:** Create *site-generalizable*, *diagnosis-relevant* brain graph embeddings.

---

## Phase 1 — Contrastive Learning Framework
You will implement methods such as:

### **Option A: GraphCL (Graph Contrastive Learning)**  
Apply augmentations such as:
- ROI dropout  
- Edge perturbation  
- Subgraph extraction  
- Feature masking  

Then contrast:
- Positive pair = 2 augmented views of same subject  
- Negative pair = other subjects  

---

### **Option B: SimCLR-Style on Flattened FC Graphs**
Augmentations:
- Gaussian noise  
- Signal smoothing  
- Random threshold jitter  
- Node permutation preserving hemispheres  

---

### **Option C: Domain-Adversarial Training**
Train an encoder that:
- Predicts ASD vs Control (supervised)  
- **Cannot** predict site (via gradient reversal)

---

## Phase 2 — Evaluation Strategy
Use two setups:

### **1. Random Split**
→ Shows basic performance  
### **2. Leave-One-Site-Out (LOSO)**
→ True test of generalizability  
→ Expected to improve drastically after contrastive learning

---

## Phase 3 — New t-SNE Visualizations
After contrastive learning, expect:

### ✔ Site clusters to dissolve  
### ✔ ASD vs Control to have more separation  
### ✔ Overall embedding space more biologically meaningful

These visualizations will complete your FYP-2 report.

---

# Citation / Acknowledgments
This project uses:
- ABIDE Preprocessed dataset  
- NetworkX  
- Scikit-learn  
- Numpy / Pandas  
- matplotlib & seaborn  

---

# Final Note
FYP-1 is fully complete.  
FYP-2 now has a clear, justified direction backed by solid analysis.

