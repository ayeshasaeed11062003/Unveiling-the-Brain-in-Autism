import numpy as np
import networkx as nx
from utils_graph import matrix_to_graph, compute_hubness  # Make sure these exist in utils_graph.py

# -----------------------------
# Load adjacency matrices
# -----------------------------
matrices_file = "C:/Users/ayesh/OneDrive/Desktop/Unveiling-the-Brain-in-Autism/padded_matrices_200.npy" 
matrices = np.load(matrices_file)
print(f"[INFO] Loaded {len(matrices)} matrices.")

# -----------------------------
# Compute hubness for each graph
# -----------------------------
hubness_list = []

for i, matrix in enumerate(matrices):
    # Convert matrix to graph
    G = matrix_to_graph(matrix, threshold=0.3)
    
    # Compute hubness
    hub_scores = compute_hubness(G)
    
    hubness_list.append(hub_scores)

print(f"[INFO] Computed hubness for {len(hubness_list)} graphs.")

# -----------------------------
# Save hubness results
# -----------------------------
np.save("hubness_scores.npy", hubness_list)
print("[INFO] Hubness scores saved as 'hubness_scores.npy'.")

# Optional: inspect first graph hubness
print("[INFO] Example hubness scores for first graph:")
print(hubness_list[0])
