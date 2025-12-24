import pandas as pd
import networkx as nx


def buildGraph(pfx="./data"):
    stations = pd.read_csv(f"{pfx}/stations.csv")
    edges = pd.read_csv(f"{pfx}/edges.csv")

    nameToUID = {
        str(row["Station"]).strip().lower(): str(row["UID"]).strip().lower()
        for _, row in stations.iterrows()
    }

    G = nx.Graph()

    uidToP = {}
    for _, row in stations.iterrows():
        uid = str(row["UID"]).strip().lower()
        p = float(str(row["Number of passengers"]).replace(",", ""))
        uidToP[uid] = p
        G.add_node(
            uid,
            name=str(row["Station"]).strip().lower(),
            ridership=p,
        )

    for _, row in edges.iterrows():
        pName = str(row["station1"]).strip().lower()
        sName = str(row["station2"]).strip().lower()
        line = str(row["line"]).strip().upper()

        p = nameToUID[pName]
        s = nameToUID[sName]
        w = 0.5 * (uidToP[p] + uidToP[s])

        G.add_edge(p, s, line=line, weight=w)

    return G, stations, edges, nameToUID
