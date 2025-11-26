# compute_efficiency.py
import numpy as np
import pandas as pd
import networkx as nx
import tqdm
from scipy.stats import ttest_ind
import utils_graph

# ----------------------------
# LOAD MATRICES + PHENOTYPE
# ----------------------------

import os

BASE_DIR = r"C:\Users\ayesh\OneDrive\Desktop\Unveiling-the-Brain-in-Autism"

matrix_path = os.path.join(BASE_DIR, "padded_matrices_200.npy")
pheno_path  = os.path.join(BASE_DIR, "ABIDE_pcp", "Phenotypic_V1_0b_preprocessed1.csv")

matrices = np.load(matrix_path, allow_pickle=True)
pheno = pd.read_csv(pheno_path)
print(f"Loaded {matrices.shape[0]} matrices from {matrix_path}")

# ----------------------------
# EFFICIENCY CALCULATION
# ----------------------------

def compute_efficiency(G):
    """Compute global efficiency of a graph."""
    if len(G.nodes()) < 2:
        return np.nan
    return nx.global_efficiency(G)

eff_list = []

print("\nComputing global efficiency for all subjects...\n")

for i in tqdm.tqdm(range(len(matrices))):

    M = matrices[i]

    # convert to graph
    G = utils_graph.matrix_to_graph(M, threshold=0.3)

    if len(G.nodes()) == 0:
        eff = np.nan
    else:
        eff = compute_efficiency(G)

    label = int(pheno.iloc[i]["DX_GROUP"]) if "DX_GROUP" in pheno.columns else np.nan

    eff_list.append({
        "Subject": i,
        "Efficiency": eff,
        "Label": label
    })

eff_df = pd.DataFrame(eff_list)
eff_df = eff_df.dropna(subset=["Label"])

eff_df["Label"] = eff_df["Label"].astype(int)
eff_df["Group"] = eff_df["Label"].map({1: "ASD", 2: "Control"})

print("\nEfficiency calculation complete!")
print(eff_df.head())

# ----------------------------
# STATISTICS
# ----------------------------

asd_vals = eff_df[eff_df["Group"] == "ASD"]["Efficiency"]
ctrl_vals = eff_df[eff_df["Group"] == "Control"]["Efficiency"]

if len(asd_vals) > 1 and len(ctrl_vals) > 1:
    t_stat, p_val = ttest_ind(asd_vals, ctrl_vals, equal_var=False)
    print(f"\nGlobal Efficiency t-test: t={t_stat:.3f}, p={p_val:.4f}")
else:
    print("\nNot enough valid data for statistical test.")

# ----------------------------
# SAVE RESULTS
# ----------------------------

eff_df.to_csv("efficiency_results.csv", index=False)
print("\nSaved → efficiency_results.csv")
print(eff_df.head())

# --- BOXPLOT FOR GLOBAL EFFICIENCY ---

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure Group column exists
if "Group" not in eff_df.columns:
    eff_df["Group"] = eff_df["Label"].map({1: "ASD", 2: "Control"})

plt.figure(figsize=(8, 5))
sns.boxplot(data=eff_df, x="Group", y="Efficiency", palette="viridis")

plt.title("Global Efficiency: ASD vs Control", fontsize=14)
plt.xlabel("Group", fontsize=12)
plt.ylabel("Efficiency Score", fontsize=12)

plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()