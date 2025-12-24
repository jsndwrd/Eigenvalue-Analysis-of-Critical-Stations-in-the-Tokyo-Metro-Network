import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src.graph import buildGraph
from src.adjacency import adjacencyMatrix
from src.eigen import qrEigen, spectralRadius

pfx = "./data"

G, stations, edges, nameToUID = buildGraph(pfx)
A, nodes = adjacencyMatrix(G)

# QR eigen stuff
eigVal, eigVec = qrEigen(A)
idx = np.argsort(eigVal)[::-1]
eigVal = eigVal[idx]
eigVec = eigVec[:, idx]

# top eigenpair and set absolute val
lambdaMax = eigVal[0]
vMax = eigVec[:, 0]
if vMax[np.argmax(np.abs(vMax))] < 0:
    vMax = -vMax

# Sort eigenvector centrality, take top 10
topK = 10
topId = np.argsort(np.abs(vMax))[::-1][:topK]
topNodes = [nodes[i] for i in topId]

# Spectral radius of the full network
rhoA = spectralRadius(A)
print("Spectral radius rho(A): ",rhoA)

# Print the top stations + how much rho drops if removed
print("Top 10 most important stations (eigenvector centrality):")
deltaRho = []
labels = []

for rank, i in enumerate(topId, start=1):
    uid = nodes[i]
    node = G.nodes[uid]
    name = node["name"]
    centrality = vMax[i]

    ridership = node.get("ridership")
    degree = G.degree[uid]
    incidents = {G[u][v]["line"] for u, v in G.edges(uid)}

    H = G.copy()
    H.remove_node(uid)
    ARed, nodesRed = adjacencyMatrix(H)
    rhoRed = spectralRadius(ARed)
    dRho = rhoA - rhoRed
    deltaRho.append(dRho)
    labels.append(name)

    print(f"\n#{rank}: {name} (UID={uid})")
    print(f"Eigenvector Centrality: {centrality:.6f}")
    pct = (dRho / rhoA * 100.0) if rhoA != 0 else 0.0
    print(f"Delta Rho: {dRho:.4f} ({pct:.2f}%)")
    print(f"Avg Passengers: {ridership}")
    # print(f"Degree: {degree}")
    # print(f"Lines: {', '.join(sorted(incidents))}")

x = np.arange(len(labels))
plt.figure(figsize=(10, 5))
plt.bar(x, deltaRho, color="steelblue")
plt.xticks(x, labels, rotation=45, ha="right")
plt.ylabel("Delta rhoS")
plt.title("Reduction in spectral radius (Delta rhoS) for critical stations")
plt.tight_layout()
plt.show()

lines = {
    "G": "green",
    "M": "red",
    "H": "pink",
    "T": "purple",
    "C": "blue",
    "Y": "gold",
    "Z": "orange",
    "N": "gray",
    "F": "brown",
}

colors = [
    lines.get(G[u][v]["line"], "black")
    for u, v in G.edges()
]

plt.figure(figsize=(15, 10))

layout = nx.kamada_kawai_layout(G)

nx.draw_networkx_edges(
    G,
    layout,
    edge_color=colors,
    width=1.4,
    alpha=0.6
)

nx.draw_networkx_nodes(
    G,
    layout,
    node_color="skyblue",
    node_size=100
)


labels = {n: G.nodes[n]["name"] for n in G.nodes}
nx.draw_networkx_labels(G, layout, labels, font_size=6)

plt.title("Tokyo Metro Network (143 Stations)")
plt.axis("off")
plt.show()
