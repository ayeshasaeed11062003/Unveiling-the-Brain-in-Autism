import numpy as np
import pandas as pd
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import networkx as nx

from load_data import load_data
from utils_graph import matrix_to_graph

def compute_clustering(G):
    """
    Computes the average clustering coefficient of the brain graph.
    The clustering coefficient measures how well a node’s neighbors
    are connected to each other (local interconnectedness).
    """
    if len(G.nodes()) == 0:
        return np.nan
    
    clustering_dict = nx.clustering(G, weight='weight')
    return np.mean(list(clustering_dict.values()))

# ---------------- MAIN PIPELINE ---------------- #

matrices, labels, pheno = load_data()

clustering_list = []

print("\nComputing clustering coefficient for all 871 subjects...\n")

for i in tqdm.tqdm(range(len(matrices))):
    mat = matrices[i]

    # Convert to graph
    G = matrix_to_graph(mat, threshold=0.3)

    # Compute clustering coefficient
    clustering_score = compute_clustering(G)

    clustering_list.append({
        "Subject": i,
        "Clustering": clustering_score,
        "Label": labels[i],
        "Group": "ASD" if labels[i] == 1 else "Control"
    })

# Create DataFrame
clustering_df = pd.DataFrame(clustering_list)
clustering_df.to_csv("C:/Users/ayesh/OneDrive/Desktop/Unveiling-the-Brain-in-Autism/clustering_results.csv", index=False)

print("Clustering coefficient saved to data/clustering_results.csv")
print(clustering_df.head())

# ---------------- VISUALIZATION ---------------- #

plt.figure(figsize=(8,5))
sns.boxplot(data=clustering_df, x="Group", y="Clustering", palette="coolwarm")
plt.title("Clustering Coefficient: ASD vs Control")
plt.show()

# ---------------- STATS TEST ---------------- #

asd_vals = clustering_df[clustering_df["Group"]=="ASD"]["Clustering"]
ctrl_vals = clustering_df[clustering_df["Group"]=="Control"]["Clustering"]

t, p = ttest_ind(asd_vals, ctrl_vals, equal_var=False)
print(f"Clustering coefficient t-test: t={t:.3f}, p={p:.4f}")
