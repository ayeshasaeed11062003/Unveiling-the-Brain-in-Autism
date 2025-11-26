import networkx as nx
import numpy as np

def matrix_to_graph(matrix, threshold=0.3):
    """
    Convert an adjacency matrix to a NetworkX graph.
    Removes self-loops and applies threshold.
    """
    adj = (matrix > threshold).astype(int)
    G = nx.from_numpy_array(adj)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def compute_hubness(graph):
    """
    Compute hubness for each node: degree centrality.
    Returns a dictionary {node: hub_score}.
    """
    hub_scores = nx.degree_centrality(graph)
    return hub_scores
