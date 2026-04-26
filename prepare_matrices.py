"""
prepare_matrices.py — Unveiling the Brain in Autism
=====================================================
Reads raw BOLD time-series files (_rois_cc200.1D, shape T×200) and
computes PEARSON CORRELATION matrices, then saves as padded_matrices_200.npy.

This is the correct preprocessing step that Bazay 2024 uses.
Run this ONCE before baseline_models.py.

Usage:
    python prepare_matrices.py
"""

import numpy as np
import os
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────
RAW_DIR     = Path(r"C:\Users\ayesh\OneDrive\Desktop\FYP\Unveiling-the-Brain-in-Autism\ABIDE_pcp\cpac\nofilt_noglobal")
OUT_FILE    = Path(r"C:\Users\ayesh\OneDrive\Desktop\FYP\Unveiling-the-Brain-in-Autism\padded_matrices_200.npy")
TARGET_ROIS = 200
MIN_TRS     = 50      # discard subjects with fewer than this many time points
# ─────────────────────────────────────────────────────────────


def ts_to_pearson(path: Path) -> np.ndarray | None:
    """
    Load a (T × R) time-series file, return (200, 200) Pearson correlation matrix.
    
    Steps:
      1. Load raw BOLD signal  →  shape (T, R)
      2. Demean + unit-variance each ROI column
      3. np.corrcoef(ts.T)     →  (R, R) Pearson r
      4. Zero the diagonal     →  remove self-connectivity
      5. Clip to [-1, 1]       →  numerical safety
    """
    try:
        ts = np.loadtxt(path)                          # (T, R)
    except Exception as e:
        print(f"  [SKIP] Cannot read {path.name}: {e}")
        return None

    if ts.ndim == 1:
        print(f"  [SKIP] {path.name}: degenerate 1D array")
        return None

    # Ensure (T, R) orientation
    if ts.shape[0] < ts.shape[1]:
        ts = ts.T

    T, R = ts.shape

    if T < MIN_TRS:
        print(f"  [SKIP] {path.name}: only {T} TRs (need ≥ {MIN_TRS})")
        return None

    # Standardise each ROI (zero mean, unit std)
    ts = ts - ts.mean(axis=0, keepdims=True)
    std = ts.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1e-8
    ts = ts / std

    # Pearson correlation
    corr = np.corrcoef(ts.T)                           # (R, R)
    np.fill_diagonal(corr, 0)
    corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)

    # Pad or trim to TARGET_ROIS × TARGET_ROIS
    if R != TARGET_ROIS:
        full = np.zeros((TARGET_ROIS, TARGET_ROIS))
        r = min(R, TARGET_ROIS)
        full[:r, :r] = corr[:r, :r]
        corr = full

    return corr


# ── MAIN ─────────────────────────────────────────────────────
files = sorted(RAW_DIR.glob("*_rois_cc200.1D"))
print(f"Found {len(files)} _rois_cc200.1D files in:\n  {RAW_DIR}\n")

if len(files) == 0:
    print("ERROR: No files found. Check RAW_DIR path above.")
    exit(1)

matrices   = []
kept_files = []

for i, fp in enumerate(files):
    corr = ts_to_pearson(fp)
    if corr is not None:
        matrices.append(corr)
        kept_files.append(fp.name)
    if (i+1) % 50 == 0:
        print(f"  Processed {i+1}/{len(files)} ...")

matrices = np.array(matrices)   # (N, 200, 200) — Pearson correlations
print(f"\nKept     : {len(matrices)} / {len(files)} subjects")
print(f"Shape    : {matrices.shape}")
print(f"Value range: {matrices.min():.4f}  to  {matrices.max():.4f}")
print(f"  (should be in [-1, 1] — if so, Pearson FC is correct)")

np.save(OUT_FILE, matrices)
print(f"\nSaved → {OUT_FILE}")
print("\nNow run: python baseline_models.py")
