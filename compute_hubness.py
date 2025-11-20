import numpy as np
import pandas as pd
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

from load_data import load_data
from utils_graph import matrix_to_graph, compute_hubness

PRINT_EDGES = False

matrices, labels, pheno = load_data()

hubness_list = []

print("🔍 Computing hubness for all 871 subjects...\n")

for i in tqdm.tqdm(range(len(matrices))):
    mat = matrices[i]

    # convert to graph
    G = matrix_to_graph(mat, threshold=0.3)
    hub_score = compute_hubness(G)

    hubness_list.append({
        "Subject": i,
        "Hubness": hub_score,
        "Label": labels[i],
        "Group": "ASD" if labels[i] == 1 else "Control"
    })

hubness_df = pd.DataFrame(hubness_list)
hubness_df.to_csv("C:\\Users\\ayesh\\OneDrive\\Desktop\\Unveiling-the-Brain-in-Autism\\hubness_results.csv", index=False)

print("Saved to hubness_results.csv")
print(hubness_df.head())

# ----- Visualization -----
plt.figure(figsize=(8,5))
sns.boxplot(data=hubness_df, x="Group", y="Hubness", palette="coolwarm")
plt.title("Hubness Comparison: ASD vs Control")
plt.show()

# ----- Statistical Test -----
asd = hubness_df[hubness_df["Group"]=="ASD"]["Hubness"]
ctrl = hubness_df[hubness_df["Group"]=="Control"]["Hubness"]

t,p = ttest_ind(asd,ctrl, equal_var=False)
print(f"T-test: t={t:.3f}, p={p:.4f}")