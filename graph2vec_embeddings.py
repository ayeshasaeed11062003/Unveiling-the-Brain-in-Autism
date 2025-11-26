#!/usr/bin/env python3
# graph2vec_embeddings.py
# Robust, venv-free script: align matrices -> phenotype, build simple graph embeddings, t-SNE, site checks.

import os
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# -----------------------------
# Config — CHANGE IF NEEDED
# -----------------------------
MATRICES_PATH = r"C:\Users\ayesh\OneDrive\Desktop\Unveiling-the-Brain-in-Autism\padded_matrices_200.npy"
RAW_DIR = r"C:\Users\ayesh\OneDrive\Desktop\Unveiling-the-Brain-in-Autism\ABIDE_pcp\cpac\nofilt_noglobal"
PHENO_PATH = r"C:\Users\ayesh\OneDrive\Desktop\Unveiling-the-Brain-in-Autism\ABIDE_pcp\Phenotypic_V1_0b_preprocessed1.csv"

THRESHOLD = 0.3   # graph threshold
TSNE_PERPLEXITY = 30

# -----------------------------
# Step 1: Load matrices (must correspond to sorted filenames)
# -----------------------------
print("[INFO] Loading padded matrices...")
matrices = np.load(MATRICES_PATH, allow_pickle=True)
n_matrices = len(matrices)
print(f"[INFO] Loaded {n_matrices} matrices.")

# -----------------------------
# Step 2: Recreate sorted filename list (must match prepare_matrices.py behavior)
# -----------------------------
print("[INFO] Reading raw filenames from folder (sorted)...")
files_all = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(("1D", ".csv", ".txt"))])
print(f"[INFO] Found {len(files_all)} raw files in {RAW_DIR}")

if len(files_all) != n_matrices:
    print("[WARNING] Number of files in RAW_DIR does not equal number of matrices.")
    print(f"  files in RAW_DIR: {len(files_all)}   matrices loaded: {n_matrices}")
    # still continue — we'll pair by index where possible

# -----------------------------
# Step 3: Extract numeric token(s) from each filename
# -----------------------------
def extract_numeric_tokens(fname):
    # returns all digit groups found in the filename as strings, e.g. 'UCLA_2_0051304...' -> ['2','0051304']
    return re.findall(r"\d+", fname)

print("[INFO] Extracting numeric tokens from filenames...")
file_tokens = [extract_numeric_tokens(f) for f in files_all]

# Build candidate ID string for each file (join or choose longest numeric token)
def candidate_from_tokens(tokens):
    if not tokens:
        return None
    # choose the longest numeric token (e.g. '0051304' over '2')
    tokens_sorted = sorted(tokens, key=lambda x: -len(x))
    return tokens_sorted[0]

file_id_raw = [candidate_from_tokens(toks) for toks in file_tokens]
print(f"[INFO] Example file -> raw id pairs (first 10):")
for i, f in enumerate(files_all[:10]):
    print(i, f, "=>", file_id_raw[i])

# -----------------------------
# Step 4: Load phenotype CSV and prepare SUB_ID strings
# -----------------------------
print("[INFO] Loading phenotype CSV...")
pheno = pd.read_csv(PHENO_PATH, dtype=str)  # load everything as str to avoid int/leading-zero mismatch
if "SUB_ID" not in pheno.columns:
    # try common alternate names
    possible = [c for c in pheno.columns if "SUB" in c.upper() or "SUB_ID" in c.upper()]
    print("[WARNING] SUB_ID column not found. Columns:", pheno.columns.tolist())
    raise SystemExit("Please ensure phenotype CSV contains 'SUB_ID' column.")

pheno["SUB_ID"] = pheno["SUB_ID"].astype(str).str.strip()
pheno_index = set(pheno["SUB_ID"].tolist())
print(f"[INFO] Phenotype file contains {len(pheno)} rows and {len(pheno_index)} unique SUB_IDs.")

# -----------------------------
# Step 5: Try multiple heuristics to match file IDs -> pheno SUB_IDs
# -----------------------------
def generate_id_candidates(raw_id):
    """
    Given a raw numeric string like '0051304' return candidate SUB_ID strings
    that might match pheno SUB_ID formatting.
    """
    if raw_id is None:
        return []
    candidates = set()
    s = raw_id
    candidates.add(s)
    # remove leading zeros
    try:
        si = str(int(s))
        candidates.add(si)
    except Exception:
        pass
    # last 5 and last 6 digits (sometimes SUB_ID is shorter)
    if len(s) >= 5:
        candidates.add(s[-5:])
    if len(s) >= 6:
        candidates.add(s[-6:])
    # with and without leading zeros trimmed to 5/6 length
    if len(s) > 5 and s.lstrip("0"):
        candidates.add(s.lstrip("0")[-5:])
    return [c for c in candidates]

# Perform matching
print("[INFO] Matching file IDs to phenotype SUB_IDs using heuristics...")
matched_indices = []     # indices of files_all that matched
matched_subids = []      # SUB_IDs matched (string)
no_match_files = []

for idx, raw in enumerate(file_id_raw):
    if raw is None:
        no_match_files.append((idx, files_all[idx]))
        continue
    found = None
    for cand in generate_id_candidates(raw):
        if cand in pheno_index:
            found = cand
            break
    if found is not None:
        matched_indices.append(idx)
        matched_subids.append(found)
    else:
        no_match_files.append((idx, files_all[idx], raw))

print(f"[INFO] Matched {len(matched_indices)} / {len(files_all)} files to phenotype SUB_IDs.")
if no_match_files:
    print("[INFO] Examples of unmatched files (up to 10):")
    for item in no_match_files[:10]:
        print("  ", item)

# -----------------------------
# Step 6: Filter matrices & filenames to matched ones (preserves original sorted order)
# -----------------------------
if len(matched_indices) == 0:
    raise SystemExit("No matched subjects found. Cannot proceed until matching is fixed.")

# Keep only those indices in the same order as matched_indices
matrices_aligned = np.array([matrices[i] for i in matched_indices])
filenames_aligned = [files_all[i] for i in matched_indices]
subids_aligned = matched_subids  # same order

print(f"[INFO] Aligned matrices: {matrices_aligned.shape[0]}")

# -----------------------------
# Step 7: Build final_labels and site labels aligned to matrices_aligned
# -----------------------------
# Build mapping from SUB_ID -> row(s) in pheno (take first match)
pheno_map = pheno.set_index("SUB_ID").to_dict(orient="index")

final_labels = []
site_labels = []
for sid in subids_aligned:
    row = pheno_map.get(sid)
    if row is None:
        final_labels.append(None)
        site_labels.append(None)
    else:
        # DX_GROUP expected to be present
        dx = row.get("DX_GROUP", None)
        site = row.get("SITE_ID", None) or row.get("SITE", None)
        # make dx numeric if string
        try:
            dx_int = int(str(dx).strip())
        except:
            dx_int = None
        final_labels.append(dx_int)
        site_labels.append(site)

# Filter out entries with missing DX_GROUP
valid_mask = [ (lbl is not None) for lbl in final_labels ]
n_valid = sum(valid_mask)
print(f"[INFO] Subjects with a valid DX_GROUP: {n_valid} / {len(final_labels)}")

if n_valid == 0:
    raise SystemExit("No aligned subjects have DX_GROUP labels. Check phenotype file column names.")

# Apply valid mask
matrices_final = np.array([m for m,ok in zip(matrices_aligned, valid_mask) if ok])
filenames_final = [fn for fn,ok in zip(filenames_aligned, valid_mask) if ok]
labels_final = np.array([lbl for lbl in final_labels if lbl is not None])
sites_final  = np.array([s for s,ok in zip(site_labels, valid_mask) if ok])

print(f"[INFO] Final dataset size: {matrices_final.shape[0]} subjects")

# -----------------------------
# Step 8: Convert matrices -> graphs and build embeddings (upper triangle)
# -----------------------------
print("[INFO] Converting matrices to thresholded graphs...")
graphs = []
for mat in matrices_final:
    adj = (mat > THRESHOLD).astype(int)
    G = nx.from_numpy_array(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    graphs.append(G)

print(f"[INFO] Built {len(graphs)} graphs.")

print("[INFO] Building embeddings by flattening upper-triangle of adjacency matrices...")
embeddings = []
for G in graphs:
    adj = nx.to_numpy_array(G)
    vec = adj[np.triu_indices_from(adj, k=1)]
    embeddings.append(vec)
embeddings = np.array(embeddings)
print("[INFO] Embeddings shape:", embeddings.shape)

# Save embeddings and labels
np.save(r"C:\Users\ayesh\OneDrive\Desktop\Unveiling-the-Brain-in-Autism\graph_embeddings.npy", embeddings)
np.save(r"C:\Users\ayesh\OneDrive\Desktop\Unveiling-the-Brain-in-Autism\site_labels.npy", sites_final)
np.save(r"C:\Users\ayesh\OneDrive\Desktop\Unveiling-the-Brain-in-Autism\dx_labels.npy", labels_final)
print("[INFO] Saved embeddings and label files.")

# -----------------------------
# Step 9: t-SNE visualization (diagnosis + site)
# -----------------------------
if embeddings.shape[0] < 2:
    raise SystemExit("Not enough samples for t-SNE.")

print("[INFO] Running t-SNE...")
perplex = min(TSNE_PERPLEXITY, max(2, (embeddings.shape[0] - 1) // 3))
tsne = TSNE(n_components=2, random_state=42, perplexity=perplex)
emb_2d = tsne.fit_transform(embeddings)

# t-SNE colored by diagnosis
plt.figure(figsize=(8,6))
sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels_final, palette="Set1", s=60, alpha=0.9)
plt.title("t-SNE of Graph Embeddings (by DX_GROUP)")
plt.savefig(r"C:\Users\ayesh\OneDrive\Desktop\Unveiling-the-Brain-in-Autism\tsne_by_dx.png", dpi=300)
plt.show()

# t-SNE colored by site (top sites only)
unique_sites, counts = np.unique(sites_final, return_counts=True)
top_sites = unique_sites[np.argsort(-counts)][:12]  # up to 12 major sites
site_colors = [s if s in top_sites else "OTHER" for s in sites_final]

plt.figure(figsize=(8,6))
sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=site_colors, palette="tab20", s=50, alpha=0.9)
plt.title("t-SNE of Graph Embeddings (by Site, top sites)")
plt.legend(bbox_to_anchor=(1.05,1), loc="upper left")
plt.savefig(r"C:\Users\ayesh\OneDrive\Desktop\Unveiling-the-Brain-in-Autism\tsne_by_site.png", dpi=300, bbox_inches="tight")
plt.show()

# Example graph plot
plt.figure(figsize=(6,6))
nx.draw(graphs[0], node_size=20, node_color="skyblue", edge_color="gray")
plt.title("Example graph (first subject)")
plt.savefig(r"C:\Users\ayesh\OneDrive\Desktop\Unveiling-the-Brain-in-Autism\example_graph.png", dpi=300)
plt.show()

print("[INFO] All done.")
