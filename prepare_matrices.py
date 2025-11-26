import numpy as np
import os

RAW_DIR = "C:\\Users\\ayesh\\OneDrive\\Desktop\\Unveiling-the-Brain-in-Autism\\ABIDE_pcp\\cpac\\nofilt_noglobal"   # folder containing the .1D / .csv matrices
TARGET_SIZE = 200                # we will standardize all matrices to 200×200

def load_matrix(path):
    """Loads .1D, .csv, or .txt matrices into numpy."""
    try:
        return np.loadtxt(path)
    except:
        raise ValueError(f"Cannot read matrix file: {path}")

files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(("1D", ".csv", ".txt"))])
matrices = []

print(f"Found {len(files)} raw matrices. Processing...")

for f in files:
    mat = load_matrix(os.path.join(RAW_DIR, f))
    mat = np.array(mat)
    rows, cols = mat.shape

    # --- Fix columns first ---
    if cols != TARGET_SIZE:
        if cols > TARGET_SIZE:
            mat = mat[:, :TARGET_SIZE]
        else:
            pad_cols = TARGET_SIZE - cols
            mat = np.pad(mat, ((0, 0), (0, pad_cols)), 'constant')

    # --- Fix rows ---
    if rows != TARGET_SIZE:
        if rows > TARGET_SIZE:
            mat = mat[:TARGET_SIZE, :]
        else:
            pad_rows = TARGET_SIZE - rows
            mat = np.pad(mat, ((pad_rows, 0), (0, 0)), 'constant')

    matrices.append(mat)

matrices = np.array(matrices)
np.save("C:\\Users\\ayesh\\OneDrive\\Desktop\\Unveiling-the-Brain-in-Autism\\padded_matrices_200", matrices)

print("DONE! Saved:", matrices.shape)
print("Saved to:", os.path.abspath("padded_matrices_200.npy"))
