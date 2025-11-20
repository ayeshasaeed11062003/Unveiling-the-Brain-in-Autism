import networkx as nx
import numpy as np

def matrix_to_graph(matrix, threshold=0.3):
    """Converts 200×200 connectivity matrix into a weighted graph."""
    G = nx.Graph()
    size = matrix.shape[0]

    for i in range(size):
        for j in range(i + 1, size):
            weight = matrix[i, j]
            if abs(weight) > threshold:
                G.add_edge(i, j, weight=weight)

    return G

def compute_hubness(G):
    """Hubness = node_degree × betweenness centrality."""
    if len(G.nodes()) == 0:
        return np.nan

    degree_dict = dict(G.degree(weight='weight'))
    centrality = nx.betweenness_centrality(G, weight='weight')

    hubness_vals = []
    for node in G.nodes():
        hub = degree_dict[node] * (centrality[node] + 1e-5)
        hubness_vals.append(hub)

    return np.mean(hubness_vals) if hubness_vals else np.nan