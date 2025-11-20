import pandas as pd
import numpy as np

PHENO_PATH = "C:\\Users\\ayesh\\OneDrive\\Desktop\\Unveiling-the-Brain-in-Autism\\ABIDE_pcp\\Phenotypic_V1_0b_preprocessed1.csv"
MATRIX_PATH = "C:\\Users\\ayesh\\OneDrive\\Desktop\\Unveiling-the-Brain-in-Autism\\padded_matrices_200.npy"

def load_data():
    # Load matrices
    matrices = np.load(MATRIX_PATH)

    # Load phenotypic CSV
    pheno = pd.read_csv(PHENO_PATH)

    # Extract symmetric labels (ASD=1, Control=2)
    labels = pheno["DX_GROUP"].values.astype(int)

    return matrices, labels, pheno

if __name__ == "__main__":
    mats, labs, pheno = load_data()
    print("Matrices:", mats.shape)
    print("Labels:", np.unique(labs, return_counts=True))