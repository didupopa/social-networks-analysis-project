import argparse
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg as la

from pathlib import Path

# paths
ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIST_CSV = DATA_DIR / "spotify_artists_2000_2022.csv"
COLLAB_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "partitioning-algorithms"


# helper function to build a graph from artists and collaborations
def build_graph(artists, collaborations):
    G = nx.Graph()
    for row in artists.itertuples(index=False):
        G.add_node(row.artist_id)
    for row in collaborations.itertuples(index=False):
        if row.artist_1_id != row.artist_2_id:
            G.add_edge(row.artist_1_id, row.artist_2_id, weight=float(row.shared_song_count))

    return G


# helper function to perform Leiden partitioning
def leiden_partition(G, seed):
    nodes = list(G.nodes)
    indices = {node: i for i, node in enumerate(nodes)}
    edges = [(indices[u], indices[v]) for u, v in G.edges]
    igraph_G = ig.Graph(n=len(nodes), edges=edges, directed=False)
    weights = [G[u][v]["weight"] for u, v in G.edges]
    partition = la.find_partition(igraph_G, la.ModularityVertexPartition, weights=weights, seed=seed, n_iterations=-1)
    return [set(nodes[i] for i in community) for community in partition]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artists", type=Path, default=ARTIST_CSV)
    parser.add_argument("--collaborations", type=Path, default=COLLAB_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    artists = pd.read_csv(args.artists, dtype={"artist_id": str})
    collaborations = pd.read_csv(args.collaborations, dtype={"artist_1_id": str, "artist_2_id": str})
    G = build_graph(artists, collaborations)
    print(f"Graph: {G.number_of_nodes()} artists, {G.number_of_edges()} collaborations")

    methods = {
        "louvain": nx.community.louvain_communities(G, weight="weight", seed=args.seed),
        "leiden": leiden_partition(G, args.seed)
    }

    assignments = []
    summary = []
    for method, groups in methods.items():
        groups = sorted(groups, key=lambda c: (-len(c), sorted(c)[0]))
        q = nx.community.modularity(G, groups, weight="weight")
        print(f"{method.capitalize()} partitioning: {len(groups)} communities, modularity={q:.4f}")
        summary.append({"algorithm": method, "communities": len(groups), "modularity": q, "largest_community": max(map(len, groups)), "seed": args.seed})
        for community_id, members in enumerate(groups, 1):
            for node in members:
                assignments.append({"artist_id": node, "algorithm": method, "community_id": community_id, "community_size": len(members)})

    pd.DataFrame(assignments).to_csv(args.output / "community_assignments.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output / "partitioning_summary.csv", index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()