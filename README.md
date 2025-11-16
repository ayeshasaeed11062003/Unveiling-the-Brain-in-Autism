# Autism Radiomics Pipeline — Hubness Feature

This project builds a radiomics feature extraction pipeline for ABIDE MRI functional connectivity matrices.

## Pipeline Overview

### 1. Convert All Matrices to 200×200
`01_prepare_matrices.py`
- Loads raw CC200 connectivity matrices of varying shapes
- Zero-pads them to a common 200×200 format
- Output: `data/padded_matrices_200.npy` (871 subjects)

### 2. Load Data & Labels
`02_load_data.py`
- Loads padded matrices
- Loads phenotypic labels (DX_GROUP)
- Returns matrices + labels + phenotypic dataframe

### 3. Graph Utilities
`03_utils_graph.py`
- Converts matrix → graph
- Computes *hubness* = node degree × betweenness centrality

### 4. Compute Hubness
`04_compute_hubness.py`
- Computes hubness for all 871 subjects
- Saves: `data/hubness_results.csv`
- Produces:
  - ASD vs Control boxplot
  - T-test statistical comparison

## Next Steps
- Add more radiomics features in separate files:
  - clustering coefficient
  - global efficiency
  - small-worldness
  - modularity
- Combine all feature files into one master dataset
