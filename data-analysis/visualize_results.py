import argparse
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

# paths
ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIST_CSV = DATA_DIR / "spotify_artists_2000_2022.csv"
COLLAB_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"
RESULTS_DIR = ROOT_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "visualizations"


# helper function to read group members from a row in the CSV
def read_group_members(row):
    return ({artist_id.strip() for artist_id in str(row).split(";") if artist_id.strip()} if pd.notna(row) else set())


# helper function to build a graph from artists and collaborations
def build_graph(artists_file, edges_file):
    artists = pd.read_csv(artists_file, dtype={"artist_id": str})
    collaborations = pd.read_csv(edges_file, dtype={"artist_1_id": str, "artist_2_id": str})
    # empty graph
    G = nx.Graph()
    for artist in artists.itertuples(index=False):
        # add artist nodes
        G.add_node(artist.artist_id, name=str(artist.artist_name))
    for collab in collaborations.itertuples(index=False):
        # only add edges for collaborations between different artists
        if collab.artist_1_id != collab.artist_2_id:
            G.add_edge(collab.artist_1_id, collab.artist_2_id, weight=int(collab.shared_song_count))
    return G


# helper function to visualize a selected subgraph of the network
# uses reproducible spring layout with node sizes based on their degree
# nodes can be colored by numerical values or highlighted to distinguish groups
# labels selected artists and saves to PNG image
def render_network(G, nodes, output, title, node_values=None, highlighted=None, label_nodes=None, seed=42):
    # draw a small induced graph with deterministic layout and legible labels
    H = G.subgraph(set(nodes) & set(G)).copy()
    if not H:
        print(f"skipping empty network {title}")
        return
    pos = nx.spring_layout(H, seed=seed, iterations=130, weight=None)
    fig, ax = plt.subplots(figsize=(12, 9))
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=.20, width=.8, edge_color="gray")
    if node_values is None:
        node_values = {n: 1 for n in H}
    values = [node_values.get(n, 0) for n in H]
    if highlighted is not None:
        palette = ["#f97316" if n in highlighted else "#cbd5e1" for n in H]
        nx.draw_networkx_nodes(H, pos, node_color=palette, node_size=[110 + 17*min(G.degree(n), 25) for n in H], linewidths=.4, edgecolors="#334155", ax=ax)
    else:
        plotted = nx.draw_networkx_nodes(H, pos, node_color=values, cmap="viridis", node_size=[100 + 14*min(G.degree(n), 35) for n in H], linewidths=.4, edgecolors="#334155", ax=ax)
        fig.colorbar(plotted, ax=ax, shrink=.7, label="score/group assignment")
    if label_nodes is None:
        label_nodes = sorted(H, key=lambda n: G.degree(n), reverse=True)[:12]
    nx.draw_networkx_labels(H, pos, labels={n: G.nodes[n].get("name", n) for n in label_nodes if n in H}, font_size=8, ax=ax)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output.name}")


# helper function to plot the 3 largest maximal cliques of the network
# read clique memberships from results/communities/cliques.csv
# generate network diagrams showing the artists in each clique
# plot distribution of maximal clique sizes across the network
def plot_cliques(G, community_folder, out):
    df = pd.read_csv(community_folder / "cliques.csv")
    # 3 largest distinct maximal cliques: complete subgraphs, all edges are present
    for i, (_, row) in enumerate(df.sort_values(["size", "total_shared_songs"], ascending=False).head(3).iterrows(), 1):
        nodes = read_group_members(row.artist_ids)
        render_network(G, nodes, out / f"clique_{i}.png",
                       f"maximal clique {i} | {len(nodes)} artists | density = 1",
                       highlighted=nodes, label_nodes=nodes)
    sizes = df["size"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(sizes.index.astype(str), sizes.values)
    ax.set(xlabel="maximal clique size (artists)", ylabel="number of maximal cliques",
           title="maximal clique size distribution")
    fig.tight_layout()
    fig.savefig(out / "clique_sizes.png", dpi=220)
    plt.close(fig)


# helper function to plot cohesive groups in independent panels, removing wasted whitespace
# selects the largest groups and chooses a suitable network layout
# based on group size and density
# highlights each group's nodes, labels selected artists
# and displays the group size and density
def plot_component_panels(G, groups, output, title, max_groups=6):
    groups = [set(g) & set(G) for g in groups if g]
    groups = [g for g in groups if g]
    groups = sorted(groups, key=len, reverse=True)[:max_groups]

    if not groups:
        return

    cols = 2 if len(groups) > 1 else 1
    rows = (len(groups) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5.5 * rows), squeeze=False)
    fig.suptitle(title, fontsize=16)

    for idx, members in enumerate(groups):
        ax = axes.flat[idx]
        H = G.subgraph(members).copy()
        if len(H) <= 20 or nx.density(H) >= 0.45:
            pos = nx.kamada_kawai_layout(H)
        else:
            pos = nx.spring_layout(H, seed=42, weight=None, iterations=150)

        nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.35, width=1.0, edge_color="gray")
        nx.draw_networkx_nodes(H, pos, ax=ax, node_color=[plt.cm.tab10(idx % 10)] * len(H), node_size=[220 + min(420, 18 * G.degree(n)) for n in H], edgecolors="#334155", linewidths=0.7)

        # label fewer nodes
        if len(H) <= 12:
            label_nodes = list(H.nodes())
        else:
            label_nodes = sorted(H, key=lambda n: G.degree(n), reverse=True)[:5]

        labels = {n: G.nodes[n].get("name", n) for n in label_nodes}
        nx.draw_networkx_labels(H, pos, labels=labels, font_size=9, ax=ax, bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=.8))
        ax.set_title(f"group {idx + 1}: {len(H)} artists, density {nx.density(H):.2f}", fontsize=11)
        ax.axis("off")
        ax.margins(0.18)

    for ax in axes.flat[len(groups):]:
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output.name}")


# helper function to plot k-core subgraphs and distribution of core numbers
# plot how the number of remaining artists decreases as k increases
# compare the sizes of the highest cumulative k-cores and visualize
# connected groups within the highest k-core
# plot the distribution of artist core numbers
def plot_kcores(G, community_folder, out, min_shell=None, max_nodes=120):
    # plot an informative stratified coreness sample and k-core retention curve
    df = pd.read_csv(community_folder / "artist_core_numbers.csv", dtype={"artist_id": str})
    coreness = {str(r.artist_id): int(r.core_number) for r in df.itertuples()}
    levels = sorted(set(coreness.values()) - {0})
    if min_shell is not None:
        levels = [k for k in levels if k >= min_shell]
    if not levels:
        print("no qualifying k-cores")
        return
    
    # retention curve uses all nodes, not the visualization sample.
    counts = [sum(v >= k for v in coreness.values()) for k in levels]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(levels, counts, marker="o", linewidth=2)
    ax.set(xlabel="k-core threshold", ylabel="artists in k-core", title="How the k-core shrinks as k increases")
    ax.set_xticks(levels)
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "kcore_retention.png", dpi=220)
    plt.close(fig)
    
    # exact coreness levels are summarized without misrepresenting the k-core hierarchy
    # the top four cumulative k-cores are shown as node-count comparisons;
    # their connected high-order regions are visualized separately below
    top_levels = sorted(levels, reverse=True)[:4]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(k) for k in top_levels], [sum(v >= k for v in coreness.values()) for k in top_levels], color="#2563eb")
    ax.set(xlabel="k-core threshold (cumulative)", ylabel="artists in k-core", title="Top four cumulative k-cores")
    fig.tight_layout()
    fig.savefig(out / "kcore_stratified_overview.png", dpi=220)
    plt.close(fig)

    # the highest k-core is small and meaningful enough to display separately.
    highest_k = max(levels)
    highest_nodes = {n for n in G if coreness.get(n, 0) >= highest_k}

    if highest_nodes:
        H_high = G.subgraph(highest_nodes).copy()
        components = sorted(nx.connected_components(H_high), key=len, reverse=True)
        plot_component_panels(G, components, out / "highest_kcore_groups.png", f"Highest {highest_k}-core: {len(highest_nodes)} artists", max_groups=6)
    
    sizes = df.core_number.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(sizes.index.astype(str), sizes.values)
    ax.set(xlabel="exact core number", ylabel="artists", title="Artist core-number distribution")
    fig.tight_layout()
    fig.savefig(out / "core_number_distribution.png", dpi=220)
    plt.close(fig)


# helper function to plot k-clique percolation communities and distribution of community sizes
# for each selected k value, display the largest community
# avoid duplicate visualizations of identical groups
# limit the displayed artists for readability and plot how community
# sizes and the number of communities change across k values
def plot_kclique_communities(G, community_folder, out, k=None, max_nodes=75):
    df = pd.read_csv(community_folder / "k_clique_communities.csv")
    levels = [k] if k is not None else sorted(df["k"].dropna().astype(int).unique())
    previous = set()
    summary = []
    for level in levels:
        frame = df[df.k == level].sort_values("size", ascending=False)
        if frame.empty:
            continue
        group = read_group_members(frame.iloc[0].artist_ids) & set(G)
        summary.append((int(level), len(frame), len(group)))
        if frozenset(group) in previous:
            print(f"k={level}: largest group already shown at another k; omitting duplicate plot")
            continue
        previous.add(frozenset(group))
        selected = sorted(group, key=lambda n: (-G.degree(n), n))[:max_nodes]
        render_network(G, selected, out / f"k_clique_community_k{level}.png", f"Largest {level}-clique percolation community ({len(group)} artists; {len(selected)} shown)", highlighted=group, label_nodes=selected if len(selected) <= 15 else selected[:8])
    if summary:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = [r[0] for r in summary]
        ax.plot(x, [r[2] for r in summary], marker="o", label="Largest community (artists)")
        ax.plot(x, [r[1] for r in summary], marker="s", label="Number of communities")
        ax.set(xlabel="k-clique threshold", ylabel="Count", title="Clique-percolation communities by k")
        ax.set_xticks(x)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "kclique_summary.png", dpi=220)
        plt.close(fig)


# visualize exact k-components computed within the highest k-core subgraph
# display the strongest k-components as separate network diagrams
# plot the number of k-components at each connectivity level
# and the size of the largest component as k increases
def plot_kcomponents_highest_core(G, components_file, out):
    if not components_file.exists():
        print("k-component CSV missing; skipping these plots")
        return
    df = pd.read_csv(components_file)
    required = {"k", "component_id", "size", "artist_ids"}
    if df.empty or not required.issubset(df.columns):
        print("k-component CSV empty or missing columns")
        return
    df = df.dropna(subset=["k", "artist_ids"]).copy()
    df["k"] = df.k.astype(int)
    df["members"] = df.artist_ids.apply(read_group_members)
    context = set().union(*df.members.tolist()) & set(G)
    if not context:
        return
    max_k = int(df.k.max())
    strongest = sorted((g for g in df[df.k == max_k].members), key=lambda g: -len(g))
    # plot each high-connectivity group distinctly, once rather than once at each k.
    plot_component_panels(G, strongest, out / "k_components_strongest_groups.png", f"Strongest k-components (k={max_k}) within highest k-core")
    counts = df.groupby("k").size()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set(xlabel="Vertex connectivity threshold k", ylabel="Number of groups", title="k-components within the highest k-core")
    fig.tight_layout()
    fig.savefig(out / "k_component_counts_highest_core.png", dpi=220)
    plt.close(fig)
    sizes = df.groupby("k")["size"].max()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes.index, sizes.values, marker="o")
    ax.set(xlabel="k", ylabel="Artists in largest k-component", title="Largest k-component size by connectivity level")
    ax.set_xticks(sizes.index)
    fig.tight_layout()
    fig.savefig(out / "k_component_largest_size.png", dpi=220)
    plt.close(fig)


# helper function to plot Louvain and Leiden community assignments and distribution of community sizes
# select the largest communities and a sample of their highest degree artists
# display communities using different colors and spatially separated layouts
# compare the community size distributions of both algorithms
def plot_partitions(G, assignments_file, out, communities_to_show=5, per_community=24):
    if not assignments_file.exists():
        print("partition CSV missing; skipping Louvain/Leiden plots")
        return
    df = pd.read_csv(assignments_file, dtype={"artist_id": str})
    algs = sorted(df.algorithm.dropna().unique())
    selected_by_alg = {}
    for algorithm in algs:
        frame = df[df.algorithm == algorithm]
        sizes = frame.groupby("community_id").size().sort_values(ascending=False)
        chosen = {}
        for community_id in sizes.index[:communities_to_show]:
            members = set(frame.loc[frame.community_id == community_id, "artist_id"]) & set(G)
            chosen[community_id] = sorted(members, key=lambda n: (-G.degree(n), n))[:per_community]
        selected_by_alg[algorithm] = chosen

    for algorithm, groups in selected_by_alg.items():
        fig, ax = plt.subplots(figsize=(14, 10))
        palette = plt.get_cmap("tab10")
        selected = set().union(*(set(v) for v in groups.values()))
        H = G.subgraph(selected)
        # spread groups by community while retaining local positions, for readability.
        centers = nx.circular_layout(range(len(groups)), scale=3.8)
        positions = {}
        for idx, (_, members) in enumerate(groups.items()):
            center = centers[idx]
            local = nx.spring_layout(G.subgraph(members), seed=42, weight=None, iterations=110)
            for n in members:
                positions[n] = center + .85 * np.asarray(local[n])
        nx.draw_networkx_edges(H, positions, ax=ax, edge_color="#aab5c1", alpha=.16, width=.65)
        for idx, (cid, members) in enumerate(groups.items()):
            nodes = [n for n in members if n in H]
            nx.draw_networkx_nodes(H, positions, nodelist=nodes, ax=ax, node_color=[palette(idx % 10)], node_size=[100 + 6 * min(G.degree(n), 40) for n in nodes], linewidths=.4, edgecolors="#334155", label=f"Community {cid} ({len(members)} shown)")
            key_nodes = sorted(nodes, key=lambda n: (-G.degree(n), n))[:2]
            nx.draw_networkx_labels(H, positions, labels={n: G.nodes[n]["name"] for n in key_nodes}, font_size=8, ax=ax)
        # color = community; size = degree; edges = collaborations
        ax.set_title(f"{algorithm.title()}: top {len(groups)} communities (up to {per_community} artists each)")
        ax.axis("off")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"{algorithm}_community_groups.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
    sizes_df = df.groupby(["algorithm", "community_id"]).size().reset_index(name="size")
    fig, ax = plt.subplots(figsize=(10, 6))
    for alg, frame in sizes_df.groupby("algorithm"):
        values = sorted(frame["size"], reverse=True)
        ax.plot(range(1, len(values)+1), values, marker=".", label=alg)
    ax.set(xlabel="community rank", ylabel="artists", title="Louvain vs Leiden community sizes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "partition_sizes.png", dpi=220)
    plt.close(fig)


# helper function to plot potential bridge artists and their neighbors
# rank artists by the number of external communities they connect to
# using betweenness centrality as an additional ranking criterion
# highlight the selected bridge candidates and display some of their
# highest degree neighbors to illustrate their network connections
def plot_bridges(G, bridges_file, out, k=None, top_n=15):
    if not bridges_file.exists():
        print("bridge assignments absent; skipping bridge plots")
        return
    df = pd.read_csv(bridges_file, dtype={"artist_id": str})
    k_values = [k] if k is not None else sorted(df["k"].dropna().astype(int).unique())
    for level in k_values:
        top = df[df.k == level].sort_values(["external_communities_reached", "betweenness_centrality"], ascending=False).head(top_n)
        if top.empty:
            print(f"no bridge candidates found at k={level}; skipping")
            continue
        bridge_nodes = set(top.artist_id)
        # limit neighbors so the picture is readble
        selected = set(bridge_nodes)
        for n in bridge_nodes:
            selected.update(sorted(G.neighbors(n), key=lambda x: G.degree(x), reverse=True)[:5])
        render_network(G, selected, out / f"bridge_artists_k{level}.png", f"potential bridge artists at k={level} (orange = ranked bridge candidates)", highlighted=bridge_nodes, label_nodes=bridge_nodes)


# helper function to plot market homophily results
# select communities with sufficient eligible collaboration edges
# and plot their average Jaccard similarity of spotify market availability
# focus on the communities with the greatest data coverage
def plot_market(homophily_file, out):
    if not homophily_file.exists():
        print("homophily data absent; skipping market homophily plots")
        return
    df = pd.read_csv(homophily_file)
    df = df[(df.market_eligible_edges >= 5) & df.mean_market_jaccard.notna()].copy()
    if df.empty:
        print("no enough eligible market edges to draw per community market chart")
        return
    # explicitly show only groups with nontrivial coverage
    df = df.sort_values("market_eligible_edges", ascending=False).head(12)
    labels = [f"{r.algorithm} {r.community_id}" for r in df.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], df.mean_market_jaccard.iloc[::-1])
    ax.set_xlim(0, 1.02)
    ax.set(xlabel="mean jaccard similarity of Spotify available market sets", ylabel="communities", title="market availability similarity in communities (>= 5 elibible edges)")
    fig.tight_layout()
    fig.savefig(out / "market_homophily.png", dpi=220)
    plt.close(fig)


# helper function to plot Girvan-Newman community assignments and modularity by cut
# display the sampled network with different colors for each community
# use separated layouts to improve readability and label key artists
# plot weighted and unweighted modularity across successive cuts
def plot_girvan_newman(G, partition_dir, out):
    path = partition_dir / "girvan_newman_assignments.csv"
    if not path.exists():
        print("Girvan-Newman assignments missing; skipping GN network")
    else:
        df = pd.read_csv(path, dtype={"artist_id": str})
        groups = []
        for cid, frame in df.groupby("community_id"):
            group = set(frame.artist_id) & set(G)
            if group:
                groups.append((cid, group))
        if groups:
            groups.sort(key=lambda item: (-len(item[1]), str(item[0])))
            selected = set().union(*(group for _, group in groups))
            H = G.subgraph(selected)
            fig, ax = plt.subplots(figsize=(14, 10))
            cmap = plt.get_cmap("tab20")
            positions = {}
            centers = nx.circular_layout(range(len(groups)), scale=4)
            for i, (_, members) in enumerate(groups):
                local = nx.spring_layout(G.subgraph(members), seed=42, weight=None)
                for node in members:
                    positions[node] = centers[i] + .9*np.asarray(local[node])
            nx.draw_networkx_edges(H, positions, ax=ax, alpha=.15, edge_color="#94a3b8", width=.6)
            for i, (cid, members) in enumerate(groups):
                nodes = sorted(members)
                nx.draw_networkx_nodes(H, positions, nodelist=nodes, ax=ax, node_color=[cmap(i % 20)]*len(nodes), node_size=[100+6*min(G.degree(n),40) for n in nodes], edgecolors="#334155", linewidths=.35, label=f"C{cid}: {len(nodes)} artists")
                key = sorted(nodes, key=lambda n: -G.degree(n))[:min(2,len(nodes))]
                nx.draw_networkx_labels(H, positions, labels={n:G.nodes[n].get("name", n) for n in key}, font_size=8, bbox=dict(facecolor="white", alpha=.8, edgecolor="none"), ax=ax)
            # colors indicate assigned communities; positions are illustrative
            ax.set_title(f"Girvan-Newman: {len(groups)} communities on sampled subgraph ({len(selected)} artists)")
            ax.axis("off")
            ax.legend(loc="center left", bbox_to_anchor=(1.0,.5), fontsize=8, frameon=False)
            fig.tight_layout()
            fig.savefig(out / "girvan_newman_community_groups.png", dpi=220, bbox_inches="tight")
            plt.close(fig)
    cuts = partition_dir / "girvan_newman_modularity_by_cut.csv"
    if cuts.exists():
        df = pd.read_csv(cuts)
        if not df.empty and "communities" in df:
            fig, ax = plt.subplots(figsize=(9,5))
            for col, label in [("modularity_unweighted","Unweighted Q"), ("modularity_weighted","Weighted Q")]:
                if col in df.columns:
                    ax.plot(df.communities, df[col], marker="o", label=label)
            if "modularity_unweighted" not in df and "modularity" in df:
                ax.plot(df.communities, df.modularity, marker="o", label="Q")
            ax.set(xlabel="number of communities after cut", ylabel="modularity", title="Girvan-Newman modularity by cut (sampled subgraph)")
            ax.axhline(0,color="gray", linewidth=.8)
            ax.grid(alpha=.2)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / "girvan_newman_modularity_by_cut.png", dpi=220)
            plt.close(fig)


# helper function to plot partitioning metrics and convergence of Louvain/Leiden algorithms
# plot weighted modularity scores while identifying differences in graph scope
# generate separate convergence plots for Louvain and Leiden
# showing modularity and optimization gain at each step
def plot_partition_metrics(partition_dir, out):
    path = partition_dir / "partitioning_summary.csv"
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty and {"algorithm", "modularity"}.issubset(df):
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(df))
            bars = ax.bar(x, df.modularity, color=["#2563eb" if a=="louvain" else "#16a34a" if a=="leiden" else "#f59e0b" for a in df.algorithm])
            labels = [f"{r.algorithm}\n({r.scope if 'scope' in df else 'network'})" for r in df.itertuples()]
            ax.set_xticks(x, labels, fontsize=9)
            ax.set_ylabel("weighted modularity Q")
            ax.set_title("Partition modularity (different scopes are NOT directly comparable)")
            ax.axhline(0,color="gray", linewidth=.8)
            for rect, q in zip(bars, df.modularity):
                ax.annotate(f"{q:.3f}", (rect.get_x()+rect.get_width()/2, rect.get_height()), ha="center", va="bottom", fontsize=9)
            fig.tight_layout()
            fig.savefig(out/"partition_modularity_comparison.png", dpi=220)
            plt.close(fig)
    path = partition_dir / "modularity_convergence.csv"
    if not path.exists():
        print("convergence CSV missing; skipping Louvain/Leiden convergence plots")
        return
    df = pd.read_csv(path)
    required = {"algorithm", "step", "modularity"}
    if df.empty or not required.issubset(df):
        return
    for algorithm, frame in df.groupby("algorithm"):
        frame = frame.sort_values("step")
        fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        axes[0].plot(frame.step, frame.modularity, marker="o", color="#2563eb")
        axes[0].set(ylabel="weighted modularity", title=f"{algorithm.title()} convergence")
        if "gain" in frame:
            axes[1].plot(frame.step, frame.gain, marker="o", color="#d97706")
            axes[1].set_ylabel("modularity gain / optimizer gain")
        axes[1].set_xlabel("hierarchical levels" if algorithm.lower()=="louvain" else "optimizer iterations")
        for ax in axes: ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out/f"{algorithm}_modularity_convergence.png", dpi=220)
        plt.close(fig)


# helper function to plot homophily results for US market availability and genre agreement
# compare US market availability between collaborating artists
# and evaluate observed agreement against a random label baseline
# visualize homophily within detected communities and summarize
# collaborations between artists of the same or different genres
def plot_homophily_additions(hom_dir, out):
    # existing market jaccard result is separate from US binary availability
    us = hom_dir / "us_market_homophily.csv"
    if us.exists():
        df = pd.read_csv(us)
        if not df.empty:
            row = df.iloc[0]
            cols = [c for c in ("US_US", "US_nonUS", "nonUS_nonUS") if c in df.columns]
            if cols:
                fig, ax = plt.subplots(figsize=(9, 5))
                values = [int(row[c]) for c in cols]
                bars = ax.bar(["US–US", "US–non-US", "non-US–non-US"][:len(cols)], values, color=["#16a34a", "#d97706", "#94a3b8"][:len(cols)])
                ax.bar_label(bars, padding=3)
                ax.set(ylabel="eligible collaboration edges", title="US market availability among collaborating artists")
                fig.tight_layout()
                fig.savefig(out/"us_market_edge_types.png", dpi=220)
                plt.close(fig)
            obs = row.get("same_US_status_rate", np.nan)
            null = row.get("null_mean_same_status_rate", np.nan)
            if pd.notna(obs) and pd.notna(null):
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(["Observed", "Random-label baseline"], [obs,null], color=["#2563eb", "#94a3b8"])
                ax.bar_label(bars, fmt="%.3f", padding=4)
                ax.set_ylim(0, 1.08)
                ax.set(ylabel="share of same-US-status edges", title="US market homophily vs random-label baseline")
                p = row.get("one_sided_p_value", np.nan)
                if pd.notna(p): ax.text(.98, .04, f"permutation p={p:.3f}", transform=ax.transAxes, ha="right", fontsize=9)
                fig.tight_layout()
                fig.savefig(out/"us_market_homophily_baseline.png", dpi=220)
                plt.close(fig)
    uscom = hom_dir/"us_community_homophily.csv"
    if uscom.exists():
        df = pd.read_csv(uscom)
        if {"algorithm", "community_id", "eligible_internal_edges", "same_US_status_rate"}.issubset(df.columns):
            df = df[(df.eligible_internal_edges>=3)&df.same_US_status_rate.notna()].copy()
            df = df.sort_values("eligible_internal_edges", ascending=False).head(12)
            if not df.empty:
                labels = [f"{r.algorithm} C{r.community_id} (n={r.eligible_internal_edges})" for r in df.itertuples()]
                fig, ax = plt.subplots(figsize=(10, max(5, len(df)*.48)))
                ax.barh(labels[::-1], df.same_US_status_rate.iloc[::-1], color="#2563eb")
                ax.set_xlim(0, 1.03)
                ax.set(xlabel="same US availability status / eligible internal edges", title="US availability agreement within communities (≥3 eligible edges)")
                fig.tight_layout()
                fig.savefig(out/"us_community_homophily.png", dpi=220)
                plt.close(fig)
    genre = hom_dir/"genre_homophily.csv"
    if genre.exists():
        df = pd.read_csv(genre)
        if not df.empty and {"same_genre_edges","cross_genre_edges"}.issubset(df):
            r = df.iloc[0]; fig,ax=plt.subplots(figsize=(8,5))
            bars = ax.bar(["Same genre", "Different genres"], [r.same_genre_edges,r.cross_genre_edges],color=["#16a34a","#d97706"])
            ax.bar_label(bars, padding=3)
            ax.set(ylabel="genre-eligible collaboration edges",title="genre homophily: observed collaborations")
            fig.tight_layout();fig.savefig(out/"genre_homophily_edges.png",dpi=220);plt.close(fig)
    community = hom_dir/"community_homophily.csv"
    if community.exists():
        df = pd.read_csv(community)
        for metric, eligible, fname, label in [("same_genre_pct","genre_eligible_edges","community_genre_homophily.png","same-genre collaborations (%)"), ("mean_market_jaccard","market_eligible_edges","market_homophily.png","mean available-market Jaccard")]:
            if not {"algorithm","community_id",metric,eligible}.issubset(df.columns):
                continue
            data = df[(df[eligible]>=5)&df[metric].notna()].copy()
            data = data.sort_values(eligible, ascending=False).head(12)
            if data.empty:
                continue
            labels = [f"{r.algorithm} C{r.community_id} (n={getattr(r,eligible)})" for r in data.itertuples()]
            fig, ax = plt.subplots(figsize=(10, max(5, len(data)*.5)))
            ax.barh(labels[::-1], data[metric].iloc[::-1], color="#2563eb")
            ax.set_xlim(0,100 if metric=="same_genre_pct" else 1.03)
            ax.set(xlabel=label, title=f"Community homophily ({eligible} ≥ 5)")
            fig.tight_layout()
            fig.savefig(out/fname, dpi=220)
            plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    # path to artist CSV
    p.add_argument("--artists", type=Path, default=ARTIST_CSV)
    # path to collaboration CSV
    p.add_argument("--collaborations", type=Path, default=COLLAB_CSV)
    # path to results folder
    p.add_argument("--results", type=Path, default=RESULTS_DIR)
    # path to output directory for plots
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    # k-clique size
    p.add_argument("--k-clique", type=int, default=None)
    # k-bridge size
    p.add_argument("--k-bridge", type=int, default=None)
    # minimum k-core shell to visualize
    p.add_argument('--k-core-min', type=int, default=None)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    # build the collaboration network graph from the artist and collaboration CSV files
    G = build_graph(args.artists, args.collaborations)
    folder = args.results / "communities"
    plot_cliques(G, folder, args.output)
    plot_kcores(G, folder, args.output, min_shell=args.k_core_min)
    plot_kclique_communities(G, folder, args.output, k=args.k_clique)
    plot_partitions(G, args.results / 'partitioning-algorithms' / 'community_assignments.csv', args.output)
    plot_girvan_newman(G, args.results / "partitioning-algorithms", args.output)
    plot_partition_metrics(args.results / "partitioning-algorithms", args.output)
    plot_bridges(G, folder / 'bridge_artists_all_k.csv', args.output, k=args.k_bridge)
    plot_homophily_additions(args.results / "homophily", args.output)
    plot_kcomponents_highest_core(G, folder / "k_components_highest_core.csv", args.output)    
    print('figures saved to:', args.output)


if __name__ == "__main__":
    main()