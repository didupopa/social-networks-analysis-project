import argparse
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg as la
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

# paths
ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIST_CSV = DATA_DIR / "spotify_artists_2000_2022.csv"
COLLAB_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "partitioning-algorithms"


# helper function to build a graph from artists and collaborations
def build_graph(artists, collaborations):
    # empty graph
    G = nx.Graph()
    for row in artists.itertuples(index=False):
        # add artist nodes
        G.add_node(row.artist_id)
    for row in collaborations.itertuples(index=False):
        # only add edges for collaborations between different artists
        if row.artist_1_id != row.artist_2_id:
            G.add_edge(row.artist_1_id, row.artist_2_id, weight=float(row.shared_song_count))

    return G


# helper function to perform Louvain partitioning
def louvain_partition(G, seed):
    # initialize history with the starting partition
    # initially, each node belongs to its own community
    # calculate initial weighted modularity
    # gain is NaN since no optimization has been done yet
    history = [{"algorithm": "louvain", "step": 0, "step_type": "level", "communities": G.number_of_nodes(), "modularity": nx.community.modularity(G, [{n} for n in G], weight="weight"), "gain": float("nan"), "stopped": False}]
    # store initial modularity for calculating improvements
    previous_q = history[0]["modularity"]
    # initialize final communities
    groups = None
    # run louvain and iterate through all partition levels
    # using seed for reproducibility
    for level, partition in enumerate(nx.community.louvain_partitions(G, weight="weight", seed=seed), start=1):
        # convert each detected community into set of nodes
        groups = [set(c) for c in partition]
        # calculate weighted modularity of current partition
        q = nx.community.modularity(G, groups, weight="weight")
        # record current level
        history.append({"algorithm": "louvain", "step": level, "step_type": "level", "communities": len(groups), "modularity": q, "gain": q - previous_q, "stopped": False})
        # update previous modularity for next iteration
        previous_q = q
        print(f"louvain level {level}: Q={q:.6f}, communities={len(groups)}")
    # if louvain produced not additional partition levels,
    # keep every node as its own community
    if groups is None:
        groups = [{n} for n in G]
    # mark laast recorded level as the stopping point
    history[-1]["stopped"] = True
    return groups, history


# helper function to perform Leiden partitioning
def leiden_partition(G, seed, max_iterations=100):
    # get list of all nodes
    nodes = list(G.nodes)
    # map each node to a numerical index
    indices = {node: i for i, node in enumerate(nodes)}
    # store all edges from original graph
    edge_pairs = list(G.edges)
    # convert networkx graph to igraph graph for Leiden algorithm
    igraph_G = ig.Graph(n=len(nodes), edges=[(indices[u], indices[v]) for u, v in edge_pairs], directed=False)
    # extract weight for every edge (weights are number of shared songs)
    weights = [G[u][v]["weight"] for u, v in edge_pairs]
    # create modularity based partition, algorithm optimizes this partition
    partition = la.ModularityVertexPartition(igraph_G, weights=weights)
    # initialize leiden optimizer
    optimiser = la.Optimiser()
    optimiser.set_rng_seed(seed)
    # record initial partition and its modularity
    # no optimization is performed at this stage
    history = [{"algorithm": "leiden", "step": 0, "step_type": "iteration", "communities": len(partition), "modularity": partition.quality(), "gain": float("nan"), "stopped": False}]
    # perform optimiation iterations up to the maximum allowed iterations or until no further improvement is possible
    for iteration in range(1, max_iterations + 1):
        # get improvement in modularity from the current partition
        gain = optimiser.optimise_partition(partition, n_iterations=1)
        # calculate modularity after this iteration
        q = partition.quality()
        # check whether optimization has stopped improving modularity
        done = gain <= 0
        # record current iteration and its results
        history.append({"algorithm": "leiden", "step": iteration, "step_type": "iteration", "communities": len(partition), "modularity": q, "gain": gain, "stopped": done})
        print(f"Leiden iteration {iteration}: Q={q:.6f}, gain={gain:.8g}, communities={len(partition)}")
        # stop iterating if no further improvement is found
        if done:
            break
        else:
            print("WARNING: Leiden reached --leiden-max-iterations before a non-improving iteration.")
    # convert vertex indices back to the original artist IDs
    groups = [set(nodes[i] for i in community) for community in partition]
    return groups, history


# helper function to save a convergence plot for Louvain or Leiden partitioning
def save_convergence_plot(history, output):
    fig, ax = plt.subplots(figsize=(9, 5))
    for algorithm, frame in pd.DataFrame(history).groupby("algorithm"):
        frame = frame.sort_values("step")
        # plot modularity against optimization step
        # circle markers and label each algorithm
        ax.plot(frame["step"], frame["modularity"], marker="o", linewidth=1.8, label=f"{algorithm.title()} ({frame.iloc[0]['step_type']})")
    ax.set(xlabel="Optimization step (different meaning for each algorithm)", ylabel="Weighted modularity", title="Community detection convergence")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)

# helper function to select a reproducible connected subgraph for Girvan-Newman partitioning
# choose a connected choesive sample inside a full network louvain group
# prefer a connected k-core with k >= min_core when one exists
# grow the sample by adding boundary artists having the most edges to already selected artists
def select_gn_subgraph(G, louvain_groups, max_nodes, min_core=3):
    # initialize list of candidate subgraphs
    candidates = []
    # iterate through communities detected by Louvain
    for group in louvain_groups:
        # create subgraph for the current community
        community_graph = G.subgraph(group)
        # skip communities with no edges
        if community_graph.number_of_edges() == 0:
            continue
        # calculate core number for each artist in the community
        cores = nx.core_number(community_graph)
        # iterate through core levels from the highest level down to 2
        for k in range(max(1, min_core), 1, -1):
            # select artists with core number at least k
            eligible = {n for n, c in cores.items() if c >= k}
            # skip this core level if no artists are eligible
            if not eligible:
                continue
            # find connected components among the eligible artists
            for component in nx.connected_components(community_graph.subgraph(eligible)):
                # keep only components with at least 3 artists
                if len(component) >= 3:
                    candidates.append((k, set(component)))
    # if no suitable k-core component was found
    if not candidates:
        # fall back to the largest connected portion of a Louvain community.
        for group in louvain_groups:
            for component in nx.connected_components(G.subgraph(group)):
                if len(component) >= 3:
                    candidates.append((1, set(component)))
    if not candidates:
        raise ValueError("no sufficiently large connected subgraph exists for Girvan-Newman")

    # sort candidates according to multiple priorities
    # prefer candidates large enough to fill max_nodes
    # then prefer higher core levels
    # then prefer larger usable samples
    # finally prefer candidates with more edges
    # sorted in descending order
    candidates.sort(key=lambda item: (len(item[1]) >= max_nodes, item[0], min(len(item[1]), max_nodes), G.subgraph(item[1]).number_of_edges()), reverse=True)
    # get highest ranked candidate and its core level
    core_k, candidate = candidates[0]
    # create a subgraph from the selected candidate
    H0 = G.subgraph(candidate)
    # if candidate already fits within the node limit, return it as is
    if len(candidate) <= max_nodes:
        return H0.copy(), core_k

    # select starting node with the highest degree
    # if degrees are equal, choose the smallest node ID
    root = min(candidate, key=lambda n: (-H0.degree(n), n))
    # initialize selected set with the starting node
    selected = {root}
    # initialize boundary with neighbors of the starting node
    boundary = set(H0.neighbors(root))
    # continue growing the sample until the node limit is reached
    while len(selected) < max_nodes and boundary:
        # select the boundary node with the most connections
        # to nodes already in the sample
        # break ties using total degree and then node ID
        node = min(boundary, key=lambda n: (-len(set(H0.neighbors(n)) & selected), -H0.degree(n), n))
        # add node to sample
        selected.add(node)
        # remove node from boundary
        boundary.discard(node)
        # add unselected neighbors of the chosen node to the boundary
        boundary.update(set(H0.neighbors(node)) - selected)
    return H0.subgraph(selected).copy(), core_k


# helper function to print basic network stats about the selected Girvan-Newman sample
def print_gn_diagnostics(H, core_k):
    # dictionary mapping of each artist to its degree
    degrees = dict(H.degree())
    print(f"Girvan-Newman sample: {len(H)} artists, {H.number_of_edges()} edges; selected within a Louvain community's {core_k}-core")
    print(f"density: {nx.density(H):.4f}; degree-1 artists: {sum(d == 1 for d in degrees.values())}; maximum degree: {max(degrees.values())}; average degree: {sum(degrees.values()) / len(degrees):.2f}")


# helper function to perform Girvan-Newman partitioning
# it uses unweighted edge betweenness to produce the cuts
# retain the best explored cut by unweighted modularity and report both scores
def girvan_newman_partition(H, max_communities):
    # check that sample contains at least 3 nodes and 1 edge
    if H.number_of_nodes() < 3 or H.number_of_edges() == 0:
        raise ValueError("Girvan-Newman sample must contain >=3 nodes and edges")
    # check that sample forms one connected component
    if not nx.is_connected(H):
        raise ValueError("Girvan-Newman sample must be connected")

    # initialize variable for best partition found
    best = None
    # initialize list to store info about each cut
    history = []
    # apply girvan-newman to generate successive community partitions
    # default algo removes edges based on unweighted edge betweenness
    for groups_tuple in nx.community.girvan_newman(H):
        # convert communities into sets of nodes
        groups = [set(g) for g in groups_tuple]
        # count number of communities in the current partition
        count = len(groups)
        # calculate modularity without considering edge weights
        q_unweighted = nx.community.modularity(H, groups, weight=None)
        # calculate modularity considering collab weights
        q_weighted = nx.community.modularity(H, groups, weight="weight")
        # record current number of communities, modularity scores and size of the largest community
        history.append({"communities": count, "modularity": q_unweighted, "modularity_unweighted": q_unweighted, "modularity_weighted": q_weighted, "largest_community": max(map(len, groups))})
        print(f"Girvan-Newman cut: {count} communities, unweighted Q={q_unweighted:.4f}, weighted Q={q_weighted:.4f}")
        # check whether this is the first partition
        # or whether it has better unweighted modularity
        if best is None or q_unweighted > best[0]:
            best = (q_unweighted, groups)
        # stop once requested maximum number of communities is reached
        # last cut may exceed the limit if one step creates multiple groups
        if count >= max_communities:
            break
    return best[1], history


# helper function to create assignment records for each artist in the communities
def assignment_records(method, groups):
    # empty list to store artist assignments
    records = []
    # sort communities by size in descending order, then by smallest artist ID in ascending order
    groups = sorted(groups, key=lambda c: (-len(c), min(c)))
    # assign an ID to each community
    for community_id, members in enumerate(groups, 1):
        # iterate through artists in the current community and sort artist IDs in ascending order
        for node in sorted(members):
            records.append({"artist_id": node, "algorithm": method, "community_id": community_id, "community_size": len(members)})
    return records


# helper function to create a summary record for the partitioning results
def summary_record(method, G, groups, seed, scope):
    return {
        "algorithm": method,                                                        # name of the partitioning algorithm used
        "communities": len(groups),                                                 # total number of communities detected
        "modularity": nx.community.modularity(G, groups, weight="weight"),          # weighted modularity of the partition
        "modularity_unweighted": nx.community.modularity(G, groups, weight=None),   # unweighted modularity of the partition
        "largest_community": max(map(len, groups)),                                 # size of the largest community detected
        "seed": seed,                                                               # random seed used for reproducibility
        "scope": scope,                                                             # description of the graph scope (e.g., full network or subgraph)
        "nodes_analyzed": G.number_of_nodes(),                                      # total number of nodes (artists) in the graph analyzed
        "edges_analyzed": G.number_of_edges()                                       # total number of edges (collaborations) in the graph analyzed
    }


def main():
    parser = argparse.ArgumentParser()
    # path to artist CSV
    parser.add_argument("--artists", type=Path, default=ARTIST_CSV)
    # path to collaboration CSV
    parser.add_argument("--collaborations", type=Path, default=COLLAB_CSV)
    # path to output directory for results
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    # random seed for reproducibility
    parser.add_argument("--seed", type=int, default=42)
    # maximum number of iterations for the Leiden algorithm
    parser.add_argument("--leiden-max-iterations", type=int, default=100)
    # parameters for Girvan-Newman subgraph selection
    parser.add_argument("--gn-nodes", type=int, default=70)
    # maximum number of communities to detect with Girvan-Newman
    parser.add_argument("--gn-max-communities", type=int, default=6)
    # minimum core level to prefer when selecting a subgraph for Girvan-Newman
    parser.add_argument("--gn-min-core", type=int, default=3)
    # option to skip Girvan-Newman partitioning
    parser.add_argument("--skip-girvan-newman", action="store_true")
    args = parser.parse_args()
    if args.leiden_max_iterations < 1:
        parser.error("--leiden-max-iterations must be >= 1")
    if args.gn_nodes < 3 or args.gn_max_communities < 2:
        parser.error("--gn-nodes must be >=3 and --gn-max-communities >=2")
    args.output.mkdir(parents=True, exist_ok=True)

    # read data and create graph
    artists = pd.read_csv(args.artists, dtype={"artist_id": str})
    collaborations = pd.read_csv(args.collaborations, dtype={"artist_1_id": str, "artist_2_id": str})
    G = build_graph(artists, collaborations)
    print(f"graph: {G.number_of_nodes()} artists, {G.number_of_edges()} collaborations")

    # apply louvain to full network
    louvain_groups, louvain_history = louvain_partition(G, args.seed)
    # applt leiden to full network
    leiden_groups, leiden_history = leiden_partition(G, args.seed, args.leiden_max_iterations)

    # store final partitions of both algorithms
    methods = {"louvain": louvain_groups, "leiden": leiden_groups}
    histories = louvain_history + leiden_history
    # save to CSV
    pd.DataFrame(histories).to_csv(args.output / "modularity_convergence.csv", index=False)
    save_convergence_plot(histories, args.output / "modularity_convergence.png")
    print("louvain completed levels:", len(louvain_history) - 1)
    print("leiden completed iterations:", len(leiden_history) - 1, "(converged)" if leiden_history[-1]["stopped"] else "(iteration cap reached)")

    # empty list to store artist to community assignments
    assignments = []
    # empty list to store summary stats
    summary = []
    # iterate through louvain and leiden results, create summary records and assignment records for each method
    for method, groups in methods.items():
        summary.append(summary_record(method, G, groups, args.seed, "full_network"))
        assignments.extend(assignment_records(method, groups))
        print(f"{method.title()}: {len(groups)} communities, weighted modularity={summary[-1]['modularity']:.4f}")
    # save to CSV
    pd.DataFrame(assignments).to_csv(args.output / "community_assignments.csv", index=False)

    # apply Girvan-Newman to a connected subgraph of the full network
    if not args.skip_girvan_newman:
        # select a smaller connected subgraph from louvain communities
        # limit sample size to the requested maximum or full graph size
        # prefer nodes belonging to the specified minimum k-core
        H, selected_core = select_gn_subgraph(G, methods["louvain"], min(args.gn_nodes, G.number_of_nodes()), args.gn_min_core)
        # print basic stats about the selected subgraph
        print_gn_diagnostics(H, selected_core)
        # run Girvan-Newman on the selected subgraph
        # store best partition and info about explored cuts
        gn_groups, history = girvan_newman_partition(H, args.gn_max_communities)
        # calculate sumary stats for girvan-newman
        # identify the analyzed graph as a louvain based k-core sample
        gn_summary = summary_record("girvan_newman", H, gn_groups, args.seed, f"louvain_connected_{selected_core}core_subgraph")
        summary.append(gn_summary)
        # save to CSV
        pd.DataFrame(assignment_records("girvan_newman", gn_groups)).to_csv(args.output / "girvan_newman_assignments.csv", index=False)
        pd.DataFrame(history).to_csv(args.output / "girvan_newman_modularity_by_cut.csv", index=False)
        print(f"selected Girvan-Newman cut: {len(gn_groups)} communities, unweighted modularity={gn_summary['modularity_unweighted']:.4f}, weighted modularity={gn_summary['modularity']:.4f}")
        print("IMPORTANT: GN modularity is measured on a smaller graph and cannot be directly compared with the full-network modularities.")

    # save to CSV
    pd.DataFrame(summary).to_csv(args.output / "partitioning_summary.csv", index=False)
    print(f"results saved to {args.output}")

if __name__ == "__main__":
    main()


# python partitioning.py --gn-nodes 100 --gn-min-core 3 --gn-max-communities 8
# graph: 2248 artists, 4997 collaborations
# louvain level 1: Q=0.645752, communities=270
# louvain level 2: Q=0.739788, communities=59
# louvain level 3: Q=0.741896, communities=46
# Leiden iteration 1: Q=0.739477, gain=0.74427017, communities=47
# Leiden iteration 2: Q=0.742540, gain=0.0030630999, communities=47
# Leiden iteration 3: Q=0.743030, gain=0.00048932123, communities=47
# Leiden iteration 4: Q=0.743269, gain=0.00023887158, communities=47
# Leiden iteration 5: Q=0.743506, gain=0.00023734466, communities=47
# Leiden iteration 6: Q=0.743537, gain=3.0934999e-05, communities=47
# Leiden iteration 7: Q=0.743638, gain=0.00010072811, communities=47
# Leiden iteration 8: Q=0.744265, gain=0.00062696575, communities=47
# Leiden iteration 9: Q=0.744413, gain=0.00014880781, communities=47
# Leiden iteration 10: Q=0.744500, gain=8.612626e-05, communities=47
# Leiden iteration 11: Q=0.744500, gain=0, communities=47
# louvain completed levels: 3
# leiden completed iterations: 11 (converged)
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