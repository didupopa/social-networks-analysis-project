import networkx as nx
import pandas as pd

from pathlib import Path
from communities import create_graph

# paths
ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIST_CSV = DATA_DIR / "spotify_artists_2000_2022.csv"
COLLAB_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "communities"

# read artis/collab data
artists = pd.read_csv(ARTIST_CSV)
collaborations = pd.read_csv(COLLAB_CSV)

# create graph from artists and collaborations
G = create_graph(artists, collaborations)

# find the highest k-core
cores = nx.core_number(G)
max_k = max(cores.values())

# extract the subgraph corresponding to the highest k-core
H = nx.k_core(G, k=max_k, core_number=cores)
print(f"highest k-core: {max_k}")
print(f"artists in selected subgraph: {H.number_of_nodes()}")

# compute exact k-components within this subgraph
components_by_k = nx.k_components(H)
# empty list to store info about each identified k-component
records = []

# iterate through the dictionary of k-components
for k, groups in sorted(components_by_k.items()):
    # iterate through identified k-components at the current level
    for i, members in enumerate(groups, 1):
        records.append({
            "k": k,                                                                     # k level of the component
            "component_id": i,                                                          # assign component identifier within current k level
            "size": len(members),                                                       # number of artists in the component
            "artist_ids": ";".join(sorted(members)),                                    # concatenate artist IDs in sorted order
            "artist_names": "; ".join(sorted(G.nodes[n]["name"] for n in members))      # concatenate artist names in sorted order
        })

# save to CSV
results = pd.DataFrame(records)
results.to_csv(OUTPUT_DIR / "k_components_highest_core.csv", index=False)
print("k-component result saved")