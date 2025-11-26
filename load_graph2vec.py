import numpy as np
import pandas as pd
import os

EMB_PATH = "graph2vec_embeddings.npy"
if not os.path.exists(EMB_PATH):
    raise FileNotFoundError(EMB_PATH)

emb = np.load(EMB_PATH)
print("Loaded embeddings:", emb.shape)

df_emb = pd.DataFrame(emb)
df_emb.to_csv("combined_embedding_features.csv", index=False)
print("Saved combined_embedding_features.csv")
