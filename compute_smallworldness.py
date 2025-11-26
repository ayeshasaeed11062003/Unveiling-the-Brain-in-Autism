import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ttest_ind
from utils_graph import matrix_to_graph  # must exist

# DEBUG: print working dir
print("CWD:", os.getcwd())

# ---------------------------
# LOAD MATRICES
# ---------------------------
MATRIX_PATH = "C:/Users/ayesh/OneDrive/Desktop/Unveiling-the-Brain-in-Autism/padded_matrices_200.npy"
if not os.path.exists(MATRIX_PATH):
    raise FileNotFoundError(f"{MATRIX_PATH} not found in {os.getcwd()}")

matrices = np.load(MATRIX_PATH, allow_pickle=True)
print(f"[DEBUG] Loaded matrices: {matrices.shape}")

if len(matrices) == 0:
    raise ValueError("[DEBUG] matrices array is empty!")

# ---------------------------
# LOAD PHENOTYPE (for labels)
# ---------------------------
PHENO_PATH = r"C:\Users\ayesh\OneDrive\Desktop\Unveiling-the-Brain-in-Autism\ABIDE_pcp\Phenotypic_V1_0b_preprocessed1.csv"
if not os.path.exists(PHENO_PATH):
    raise FileNotFoundError(f"{PHENO_PATH} not found")

pheno = pd.read_csv(PHENO_PATH)
print(f"[DEBUG] Loaded phenotype file with shape: {pheno.shape}")

if "DX_GROUP" not in pheno.columns:
    raise ValueError("DX_GROUP column missing from phenotype file.")

# Align counts
if len(pheno) < len(matrices):
    print("[WARNING] phenotype rows fewer than matrices. Truncating matrices.")
    matrices = matrices[:len(pheno)]

labels = pheno["DX_GROUP"].astype(int).values
print(f"[DEBUG] Loaded labels for {len(labels)} subjects.")
print(f"[DEBUG] Label distribution:\n{pd.Series(labels).value_counts()}")

# ---------------------------
# QUALITY CONTROL FUNCTIONS
# ---------------------------
def qc_motion(pheno_row):
    fd_mean = pheno_row.get("MeanFD", None)
    fd_max = pheno_row.get("MaxFD", None)
    fd_perc = pheno_row.get("FD_perc", None)
    if fd_mean is None:
        return True
    if fd_mean > 0.20:
        return False
    if fd_max is not None and fd_max > 0.5:
        return False
    if fd_perc is not None and fd_perc > 20:
        return False
    return True

def qc_connectivity(M):
    if np.isnan(M).any():
        return False
    if np.var(M) < 1e-4:
        return False
    if np.mean(M == 0) > 0.40:
        return False
    if np.mean(M < 0) > 0.50:
        return False
    if np.mean(np.abs(M)) < 0.05:
        return False
    return True

# ---------------------------
# Build QC mask
# ---------------------------
qc_mask = []
for i in range(len(matrices)):
    motion_ok = qc_motion(pheno.iloc[i])
    conn_ok = qc_connectivity(matrices[i])
    qc_mask.append(motion_ok and conn_ok)
qc_mask = np.array(qc_mask)
print(f"[QC] Subjects passing QC: {qc_mask.sum()} / {len(qc_mask)}")

# Save QC flags
qc_df = pd.DataFrame({"Subject": np.arange(len(qc_mask)), "QC_pass": qc_mask.astype(int)})
qc_df.to_csv("QC_flags.csv", index=False)

# ---------------------------
# SMALL-WORLDNESS FUNCTION
# ---------------------------
def compute_small_worldness(G, n_rand=8):
    if len(G.nodes()) < 5:
        return np.nan
    C = nx.average_clustering(G, weight="weight")
    try:
        # if negative weights present, fallback to unweighted path length
        edges_with_weight = list(G.edges(data="weight"))
        if any(w < 0 for (_, _, w) in edges_with_weight):
            L = nx.average_shortest_path_length(G)
        else:
            L = nx.average_shortest_path_length(G, weight="weight")
    except Exception as e:
        print(f"[DEBUG] Error computing L: {e}")
        return np.nan

    nodes, edges = G.number_of_nodes(), G.number_of_edges()
    C_rands, L_rands = [], []
    for _ in range(n_rand):
        R = nx.gnm_random_graph(nodes, edges)
        C_rands.append(nx.average_clustering(R))
        try:
            L_rands.append(nx.average_shortest_path_length(R))
        except:
            continue
    if len(L_rands) == 0:
        return np.nan
    C_rand = np.mean(C_rands)
    L_rand = np.mean(L_rands)
    if C_rand == 0 or L_rand == 0:
        return np.nan
    return (C / C_rand) / (L / L_rand)

# ---------------------------
# MAIN LOOP
# ---------------------------
sw_list = []
print("[INFO] Computing small-worldness for QC-passed subjects...")

for i in tqdm(range(len(matrices))):
    if not qc_mask[i]:
        continue
    M = np.nan_to_num(matrices[i])
    M = np.where(M < 0, 0, M)   # zero negatives
    G = matrix_to_graph(M, threshold=0.3)
    if G.number_of_nodes() == 0:
        print(f"[DEBUG] Graph {i} is empty. Skipping.")
        continue
    sw = compute_small_worldness(G)
    sw_list.append({
        "Subject": i,
        "SmallWorldness": sw,
        "Label": labels[i],
        "Group": "ASD" if labels[i] == 1 else "Control"
    })

sw_df = pd.DataFrame(sw_list)
print(f"[DEBUG] Before dropping NaNs: {sw_df.shape}")
sw_df = sw_df.dropna()
print(f"[DEBUG] After dropping NaNs: {sw_df.shape}")
print(sw_df.head())

# ---------------------------
# PLOT
# ---------------------------
if not sw_df.empty:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="Group", y="SmallWorldness", data=sw_df, palette="coolwarm")
    plt.title("Small-Worldness: ASD vs Control")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.show()
else:
    print("[WARNING] No data to plot.")

# ---------------------------
# STAT TEST
# ---------------------------
asd = sw_df[sw_df["Group"] == "ASD"]["SmallWorldness"]
ctrl = sw_df[sw_df["Group"] == "Control"]["SmallWorldness"]
if len(asd) > 0 and len(ctrl) > 0:
    t, p = ttest_ind(asd, ctrl, equal_var=False)
    print(f"[INFO] Small-Worldness t-test: t={t:.3f}, p={p:.4f}")
else:
    print("[WARNING] Not enough data for t-test.")

# ---------------------------
# SAVE
# ---------------------------
sw_df.to_csv("smallworldness_results.csv", index=False)
print("[INFO] Saved smallworldness_results.csv")
