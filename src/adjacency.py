import numpy as np
import networkx as nx

def adjacencyMatrix(G: nx.Graph):
    nodes = list(G.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)

    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        i, j = idx[u], idx[v]
        A[i, j] = w
        A[j, i] = w

    return A, nodes
