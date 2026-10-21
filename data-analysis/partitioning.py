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


# helper function to perform Louvain partitioning
def louvain_partition(G, seed):
    return nx.community.louvain_communities(G, weight="weight", seed=seed)


# helper function to perform Leiden partitioning
def leiden_partition(G, seed):
    nodes = list(G.nodes)
    indices = {node: i for i, node in enumerate(nodes)}
    edges = [(indices[u], indices[v]) for u, v in G.edges]
    igraph_G = ig.Graph(n=len(nodes), edges=edges, directed=False)
    weights = [G[u][v]["weight"] for u, v in G.edges]
    partition = la.find_partition(igraph_G, la.ModularityVertexPartition, weights=weights, seed=seed, n_iterations=-1)
    return [set(nodes[i] for i in community) for community in partition]


# helper function to select a reproducible connected subgraph for Girvan-Newman partitioning
# choose a connected choesive sample inside a full network louvain group
# prefer a connected k-core with k >= min_core when one exists
# grow the sample by adding boundary artists having the most edges to already selected artists
def select_gn_subgraph(G, louvain_groups, max_nodes, min_core=3):
    candidates = []
    for group in louvain_groups:
        community_graph = G.subgraph(group)
        if community_graph.number_of_edges() == 0:
            continue
        cores = nx.core_number(community_graph)
        for k in range(max(1, min_core), 1, -1):
            eligible = {n for n, c in cores.items() if c >= k}
            if not eligible:
                continue
            for component in nx.connected_components(community_graph.subgraph(eligible)):
                if len(component) >= 3:
                    candidates.append((k, set(component)))
    if not candidates:
        # Fall back to the largest connected portion of a Louvain community.
        for group in louvain_groups:
            for component in nx.connected_components(G.subgraph(group)):
                if len(component) >= 3:
                    candidates.append((1, set(component)))
    if not candidates:
        raise ValueError("no sufficiently large connected subgraph exists for Girvan-Newman")

    # prefer a core that can fill the requested sample; otherwise the largest
    # available cohesive component. Ties are resolved deterministically
    candidates.sort(key=lambda item: (
        len(item[1]) >= max_nodes,
        item[0],
        min(len(item[1]), max_nodes),
        G.subgraph(item[1]).number_of_edges(),
    ), reverse=True)
    core_k, candidate = candidates[0]
    H0 = G.subgraph(candidate)
    if len(candidate) <= max_nodes:
        return H0.copy(), core_k

    root = min(candidate, key=lambda n: (-H0.degree(n), n))
    selected = {root}
    boundary = set(H0.neighbors(root))
    while len(selected) < max_nodes and boundary:
        node = min(boundary, key=lambda n: (
            -len(set(H0.neighbors(n)) & selected),
            -H0.degree(n),
            n,
        ))
        selected.add(node)
        boundary.discard(node)
        boundary.update(set(H0.neighbors(node)) - selected)
    return H0.subgraph(selected).copy(), core_k


def print_gn_diagnostics(H, core_k):
    degrees = dict(H.degree())
    print(f"Girvan-Newman sample: {len(H)} artists, {H.number_of_edges()} edges; selected within a Louvain community's {core_k}-core", flush=True)
    print(f"density: {nx.density(H):.4f}; degree-1 artists: {sum(d == 1 for d in degrees.values())}; maximum degree: {max(degrees.values())}; average degree: {sum(degrees.values()) / len(degrees):.2f}", flush=True)


# helper function to perform Girvan-Newman partitioning
# it uses unweighted edge betweenness to produce the cuts
# retain the best explored cut by unweighted modularity and report both scores
def girvan_newman_partition(H, max_communities):
    if H.number_of_nodes() < 3 or H.number_of_edges() == 0:
        raise ValueError("Girvan-Newman sample must contain >=3 nodes and edges")
    if not nx.is_connected(H):
        raise ValueError("Girvan-Newman sample must be connected")

    best = None
    history = []
    for groups_tuple in nx.community.girvan_newman(H):
        groups = [set(g) for g in groups_tuple]
        count = len(groups)
        q_unweighted = nx.community.modularity(H, groups, weight=None)
        q_weighted = nx.community.modularity(H, groups, weight="weight")
        history.append({"communities": count, "modularity": q_unweighted,
                        "modularity_unweighted": q_unweighted,
                        "modularity_weighted": q_weighted,
                        "largest_community": max(map(len, groups))})
        print(f"Girvan-Newman cut: {count} communities, unweighted Q={q_unweighted:.4f}, weighted Q={q_weighted:.4f}", flush=True)
        if best is None or q_unweighted > best[0]:
            best = (q_unweighted, groups)
        if count >= max_communities:
            break
    return best[1], history


# helper function to create assignment records for each artist in the communities
def assignment_records(method, groups):
    records = []
    groups = sorted(groups, key=lambda c: (-len(c), min(c)))
    for community_id, members in enumerate(groups, 1):
        for node in sorted(members):
            records.append({"artist_id": node, "algorithm": method,
                            "community_id": community_id,
                            "community_size": len(members)})
    return records


# helper function to create a summary record for the partitioning results
def summary_record(method, G, groups, seed, scope):
    return {
        "algorithm": method,
        "communities": len(groups),
        "modularity": nx.community.modularity(G, groups, weight="weight"),
        "modularity_unweighted": nx.community.modularity(G, groups, weight=None),
        "largest_community": max(map(len, groups)),
        "seed": seed,
        "scope": scope,
        "nodes_analyzed": G.number_of_nodes(),
        "edges_analyzed": G.number_of_edges()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artists", type=Path, default=ARTIST_CSV)
    parser.add_argument("--collaborations", type=Path, default=COLLAB_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gn-nodes", type=int, default=70, help="maximum nodes in the connected GN sample (default: 70)")
    parser.add_argument("--gn-max-communities", type=int, default=6, help="stop GN after reaching this many communities (default: 6)")
    parser.add_argument("--gn-min-core", type=int, default=3, help="preferred minimum core level inside Louvain groups (default: 3)")
    parser.add_argument("--skip-girvan-newman", action="store_true")
    args = parser.parse_args()
    if args.gn_nodes < 3 or args.gn_max_communities < 2:
        parser.error("--gn-nodes must be >=3 and --gn-max-communities >=2")
    args.output.mkdir(parents=True, exist_ok=True)

    artists = pd.read_csv(args.artists, dtype={"artist_id": str})
    collaborations = pd.read_csv(args.collaborations, dtype={"artist_1_id": str, "artist_2_id": str})
    G = build_graph(artists, collaborations)
    print(f"graph: {G.number_of_nodes()} artists, {G.number_of_edges()} collaborations")

    methods = {
        "louvain": nx.community.louvain_communities(G, weight="weight", seed=args.seed),
        "leiden": leiden_partition(G, args.seed)
    }

    assignments = []
    summary = []
    for method, groups in methods.items():
        summary.append(summary_record(method, G, groups, args.seed, "full_network"))
        assignments.extend(assignment_records(method, groups))
        print(f"{method.title()}: {len(groups)} communities, weighted modularity={summary[-1]['modularity']:.4f}", flush=True)
    pd.DataFrame(assignments).to_csv(args.output / "community_assignments.csv", index=False)

    if not args.skip_girvan_newman:
        H, selected_core = select_gn_subgraph(
            G, methods["louvain"], min(args.gn_nodes, G.number_of_nodes()), args.gn_min_core
        )
        print_gn_diagnostics(H, selected_core)
        gn_groups, history = girvan_newman_partition(H, args.gn_max_communities)
        gn_summary = summary_record("girvan_newman", H, gn_groups, args.seed,
                                    f"louvain_connected_{selected_core}core_subgraph")
        summary.append(gn_summary)
        pd.DataFrame(assignment_records("girvan_newman", gn_groups)).to_csv(
            args.output / "girvan_newman_assignments.csv", index=False)
        pd.DataFrame(history).to_csv(args.output / "girvan_newman_modularity_by_cut.csv", index=False)
        print(f"selected Girvan-Newman cut: {len(gn_groups)} communities, "
              f"unweighted modularity={gn_summary['modularity_unweighted']:.4f}, "
              f"weighted modularity={gn_summary['modularity']:.4f}", flush=True)
        print("IMPORTANT: GN modularity is measured on a smaller graph and cannot "
              "be directly compared with the full-network modularities.", flush=True)

    pd.DataFrame(summary).to_csv(args.output / "partitioning_summary.csv", index=False)
    print(f"results saved to {args.output}")

if __name__ == "__main__":
    main()


# python partitioning.py --gn-nodes 100 --gn-min-core 3 --gn-max-communities 8
# graph: 2248 artists, 4997 collaborations
# Louvain: 46 communities, weighted modularity=0.7419
# Leiden: 47 communities, weighted modularity=0.7445
# Girvan-Newman sample: 100 artists, 249 edges; selected within a Louvain community's 2-core
# density: 0.0503; degree-1 artists: 1; maximum degree: 53; average degree: 4.98
# Girvan-Newman cut: 2 communities, unweighted Q=0.4028, weighted Q=0.2322
# Girvan-Newman cut: 3 communities, unweighted Q=0.4913, weighted Q=0.3189
# Girvan-Newman cut: 4 communities, unweighted Q=0.5168, weighted Q=0.3510
# Girvan-Newman cut: 5 communities, unweighted Q=0.5411, weighted Q=0.3638
# Girvan-Newman cut: 6 communities, unweighted Q=0.5381, weighted Q=0.3618
# Girvan-Newman cut: 7 communities, unweighted Q=0.5372, weighted Q=0.3609
# Girvan-Newman cut: 8 communities, unweighted Q=0.5358, weighted Q=0.3598
# selected Girvan-Newman cut: 5 communities, unweighted modularity=0.5411, weighted modularity=0.3638
# IMPORTANT: GN modularity is measured on a smaller graph and cannot be directly compared with the full-network modularities.
# python partitioning.py --gn-nodes 250 --gn-min-core 3 --gn-max-communities 15
# graph: 2248 artists, 4997 collaborations
# Louvain: 46 communities, weighted modularity=0.7419
# Leiden: 47 communities, weighted modularity=0.7445
# Girvan-Newman sample: 59 artists, 176 edges; selected within a Louvain community's 3-core
# density: 0.1029; degree-1 artists: 0; maximum degree: 30; average degree: 5.97
# Girvan-Newman cut: 2 communities, unweighted Q=0.2239, weighted Q=0.1657
# Girvan-Newman cut: 3 communities, unweighted Q=0.4906, weighted Q=0.2848
# Girvan-Newman cut: 4 communities, unweighted Q=0.5048, weighted Q=0.2819
# Girvan-Newman cut: 5 communities, unweighted Q=0.5044, weighted Q=0.2748
# Girvan-Newman cut: 6 communities, unweighted Q=0.4985, weighted Q=0.2715
# Girvan-Newman cut: 7 communities, unweighted Q=0.4915, weighted Q=0.2450
# Girvan-Newman cut: 8 communities, unweighted Q=0.5198, weighted Q=0.2637
# Girvan-Newman cut: 9 communities, unweighted Q=0.5357, weighted Q=0.2679
# Girvan-Newman cut: 10 communities, unweighted Q=0.5254, weighted Q=0.2557
# Girvan-Newman cut: 11 communities, unweighted Q=0.5190, weighted Q=0.2507
# Girvan-Newman cut: 12 communities, unweighted Q=0.5073, weighted Q=0.2415
# Girvan-Newman cut: 13 communities, unweighted Q=0.5032, weighted Q=0.2441
# Girvan-Newman cut: 14 communities, unweighted Q=0.4934, weighted Q=0.2441
# Girvan-Newman cut: 15 communities, unweighted Q=0.4853, weighted Q=0.2424
# selected Girvan-Newman cut: 9 communities, unweighted modularity=0.5357, weighted modularity=0.2679
# IMPORTANT: GN modularity is measured on a smaller graph and cannot be directly compared with the full-network modularities.