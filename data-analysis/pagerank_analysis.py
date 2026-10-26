import csv
import json
from pathlib import Path
from statistics import mean
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PAGERANK_ANALYSIS_DIR = RESULTS_DIR / "pagerank"
ARTISTS_CSV = DATA_DIR / "spotify_artists_2000_2022.csv"
COLLABS_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"

# output files in the pagerank results folder
OUTPUT_NODE_CSV = RESULTS_DIR / PAGERANK_ANALYSIS_DIR / "pagerank-node-metrics.csv"
OUTPUT_SUMMARY_JSON = RESULTS_DIR / PAGERANK_ANALYSIS_DIR / "pagerank-summary.json"
OUTPUT_TOP_PLOT = RESULTS_DIR / PAGERANK_ANALYSIS_DIR / "pagerank-top-artists.png"
OUTPUT_SCATTER_PLOT = RESULTS_DIR / PAGERANK_ANALYSIS_DIR / "pagerank-vs-degree.png"

# how many top-ranked artists count as "top" for RQ2/RQ3/RQ4
TOP_N = 20

# RQ4
# csv written by the partitioning script, with louvain and leiden assignments
COMMUNITY_ASSIGNMENTS_CSV = (
    DATA_DIR.parent
    / "results"
    / "partitioning-algorithms"
    / "community_assignments.csv"
)

# build the graph, including the collab-year edge attributes
G = nx.Graph()

with ARTISTS_CSV.open("r", newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        G.add_node(row["artist_id"], name=row["artist_name"])

with COLLABS_CSV.open("r", newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        a1, a2 = row["artist_1_id"], row["artist_2_id"]
        weight = int(row["shared_song_count"])

        if a1 not in G:
            G.add_node(a1, name=row["artist_1_name"])
        if a2 not in G:
            G.add_node(a2, name=row["artist_2_name"])

        # the weight of an edge is the number of shared songs; the year attributes are only added if the csv has them
        edge_attrs = {"weight": weight}
        
        if row.get("first_collab_year"):
            edge_attrs["first_collab_year"] = int(row["first_collab_year"])
            edge_attrs["last_collab_year"] = int(row["last_collab_year"])

        G.add_edge(a1, a2, **edge_attrs)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# check whether any edge has year information, needed for rq3
has_year_data = any("first_collab_year" in d for _, _, d in G.edges(data=True))
if not has_year_data:
    print("No first_collab_year/last_collab_year found on edges -- RQ3 will be skipped.")
    print("(Rerun build-collaborations.py with the year-tracking update first.)")


# artist name of a node, falling back to its id
def name_of(node):
    return G.nodes[node].get("name", node)


# turn values into ranks; ties are ranked in arbitrary order
def rank_dict(values_dict):
    """rank 1 = highest value"""
    ordered = sorted(values_dict, key=values_dict.get, reverse=True)
    return {node: i + 1 for i, node in enumerate(ordered)}


# core metrics
degree_dict = dict(G.degree())
weighted_degree_dict = dict(G.degree(weight="weight"))

# pagerank with edge weights (shared songs) and without
pagerank_weighted = nx.pagerank(G, weight="weight")
pagerank_unweighted = nx.pagerank(G, weight=None)

# ranks for degree and for both pagerank versions
degree_rank = rank_dict(degree_dict)
pagerank_rank = rank_dict(pagerank_weighted)
pagerank_unweighted_rank = rank_dict(pagerank_unweighted)

# positive rank_diff = PageRank ranks this artist HIGHER (more important)
# than raw degree alone would suggest ("prestige over popularity")
rank_diff = {
    node: degree_rank[node] - pagerank_rank[node]
    for node in G.nodes()
}

# all results are collected in one dict that is written to json at the end
summary = {"graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges()}}

# RQ1: Does PageRank surface "prestige" that raw popularity (degree) misses?
# the 15 biggest gains and the 15 biggest drops in rank when going from degree to pagerank
overperformers = sorted(rank_diff, key=rank_diff.get, reverse=True)[:15]
underperformers = sorted(rank_diff, key=rank_diff.get)[:15]

# spearman correlation between degree and weighted pagerank over all artists
spearman_corr, spearman_p = spearmanr(
    [degree_dict[n] for n in G.nodes()],
    [pagerank_weighted[n] for n in G.nodes()],
)

summary["rq1_prestige_vs_popularity"] = {
    "description": (
        "Spearman correlation between degree and weighted PageRank; "
        "artists where PageRank rank is much better than degree rank "
        "('overperformers') have few collaborators but well-connected ones."
    ),
    "spearman_correlation": spearman_corr,
    "spearman_p_value": spearman_p,
    "top_overperformers": [
        {
            "artist": name_of(n),
            "degree": degree_dict[n],
            "degree_rank": degree_rank[n],
            "pagerank_rank": pagerank_rank[n],
            "rank_improvement": rank_diff[n],
        }
        for n in overperformers
    ],
    "top_underperformers": [
        {
            "artist": name_of(n),
            "degree": degree_dict[n],
            "degree_rank": degree_rank[n],
            "pagerank_rank": pagerank_rank[n],
            "rank_drop": -rank_diff[n],
        }
        for n in underperformers
    ],
}

# RQ2: Does edge weight (shared song count) change who ranks highest?
# top n artists by pagerank with and without weights
top_weighted = set(sorted(pagerank_weighted, key=pagerank_weighted.get, reverse=True)[:TOP_N])
top_unweighted = set(sorted(pagerank_unweighted, key=pagerank_unweighted.get, reverse=True)[:TOP_N])

# overlap of the two top lists (jaccard similarity)
overlap = top_weighted & top_unweighted
jaccard = len(overlap) / len(top_weighted | top_unweighted)

summary["rq2_weighted_vs_unweighted"] = {
    "description": (
        f"Top-{TOP_N} PageRank artists using shared-song-count as edge weight, "
        "vs. treating every collaboration edge as equally important."
    ),
    "top_n": TOP_N,
    "overlap_count": len(overlap),
    "jaccard_similarity": jaccard,
    "only_in_weighted_top": [name_of(n) for n in top_weighted - top_unweighted],
    "only_in_unweighted_top": [name_of(n) for n in top_unweighted - top_weighted],
}

# RQ3: Are top-ranked artists concentrated by era?
if has_year_data:
    # an artist's era is the mean of the midpoints of the years of all their collaborations
    def artist_era(node):
        years = []
        for _, _, data in G.edges(node, data=True):
            if "first_collab_year" in data:
                years.append((data["first_collab_year"] + data["last_collab_year"]) / 2)
        return mean(years) if years else None

    # era of the top pagerank artists and of the whole network
    top_pagerank_nodes = sorted(pagerank_weighted, key=pagerank_weighted.get, reverse=True)[:TOP_N]
    top_eras = [e for e in (artist_era(n) for n in top_pagerank_nodes) if e is not None]
    all_eras = [e for e in (artist_era(n) for n in G.nodes()) if e is not None]

    # group years into decades
    def decade_bucket(year):
        return f"{int(year) // 10 * 10}s"

    # count artists per decade for the top artists and for the whole network
    top_decade_counts = {}
    for e in top_eras:
        bucket = decade_bucket(e)
        top_decade_counts[bucket] = top_decade_counts.get(bucket, 0) + 1

    all_decade_counts = {}
    for e in all_eras:
        bucket = decade_bucket(e)
        all_decade_counts[bucket] = all_decade_counts.get(bucket, 0) + 1

    summary["rq3_era_concentration"] = {
        "description": (
            f"Mean collaboration year of each artist's edges, for the top-{TOP_N} "
            "PageRank artists vs. the whole network, bucketed by decade."
        ),
        "top_n": TOP_N,
        "mean_era_top_pagerank": mean(top_eras) if top_eras else None,
        "mean_era_whole_network": mean(all_eras) if all_eras else None,
        "top_pagerank_decade_counts": top_decade_counts,
        "whole_network_decade_counts": all_decade_counts,
    }
else:
    summary["rq3_era_concentration"] = {"skipped": "no first_collab_year/last_collab_year on edges"}


# RQ4: Do high-PageRank artists act as bridges between communities?
# Use precomputed Louvain and Leiden assignments from the specific CSV.

def load_community_mapping(csv_path, algorithm):
    """Load artist_id -> community_id for one algorithm."""
    mapping = {}

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        # the csv must have these columns; the algorithm column is optional
        required = {"artist_id", "community_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{csv_path} must contain columns: {required}"
            )

        has_algorithm_column = "algorithm" in reader.fieldnames

        for row in reader:
            if has_algorithm_column:
                # keep only the rows of the requested algorithm
                if row["algorithm"].strip().lower() != algorithm:
                    continue

            artist_id = row["artist_id"]
            community_id = row["community_id"]

            # an artist can only be in one community per algorithm
            if artist_id in mapping:
                raise ValueError(
                    f"Duplicate assignment for artist {artist_id} "
                    f"in {algorithm} data."
                )

            mapping[artist_id] = community_id

    if not mapping:
        raise ValueError(
            f"No community assignments found for {algorithm} "
            f"in {csv_path}"
        )

    # Every graph node must have an assignment.
    missing = set(G.nodes()) - set(mapping)
    if missing:
        raise ValueError(
            f"{len(missing)} graph artists have no {algorithm} "
            f"community assignment. Examples: {list(missing)[:5]}"
        )

    return mapping


def bridge_score(node, community_of):
    """Fraction of neighbors in a different community."""
    neighbors = list(G.neighbors(node))

    # artists without neighbours get a score of 0
    if not neighbors:
        return 0.0

    # count the neighbours that are in a different community
    different = sum(
        1 for neighbor in neighbors
        if community_of[neighbor] != community_of[node]
    )

    return different / len(neighbors)


# the top n artists by pagerank, and everyone else
top_pagerank_nodes = sorted(
    pagerank_weighted,
    key=pagerank_weighted.get,
    reverse=True
)[:TOP_N]

top_pagerank_set = set(top_pagerank_nodes)
other_nodes = [
    node for node in G.nodes()
    if node not in top_pagerank_set
]

rq4_results = {}
community_mappings = {}
bridge_scores_by_algorithm = {}

# for each algorithm: load the communities, compute bridge scores and compare the top artists with the rest
for algorithm in ("louvain", "leiden"):
    community_of = load_community_mapping(
        COMMUNITY_ASSIGNMENTS_CSV, algorithm
    )
    community_mappings[algorithm] = community_of

    bridge_scores = {
        node: bridge_score(node, community_of)
        for node in G.nodes()
    }
    bridge_scores_by_algorithm[algorithm] = bridge_scores

    # group artists by community to get the number and sizes of communities
    communities = {}
    for node, community_id in community_of.items():
        communities.setdefault(community_id, set()).add(node)

    top_bridge_scores = [
        bridge_scores[node] for node in top_pagerank_nodes
    ]
    other_bridge_scores = [
        bridge_scores[node] for node in other_nodes
    ]

    # correlation between pagerank and bridge score over all artists
    bridge_corr, bridge_corr_p = spearmanr(
        [pagerank_weighted[node] for node in G.nodes()],
        [bridge_scores[node] for node in G.nodes()]
    )

    # mann whitney test of the bridge scores: top artists vs the rest
    if top_bridge_scores and other_bridge_scores:
        mw_stat, mw_p = mannwhitneyu(
            top_bridge_scores,
            other_bridge_scores,
            alternative="two-sided"
        )
    else:
        mw_stat, mw_p = None, None

    rq4_results[algorithm] = {
        "description": (
            f"{algorithm.capitalize()} communities loaded from the "
            "precomputed assignments CSV. Bridge score is the fraction "
            "of an artist's neighbors in a different community. "
            f"Compares the top-{TOP_N} PageRank artists with the rest."
        ),
        "num_communities": len(communities),
        "community_sizes": sorted(
            (len(members) for members in communities.values()),
            reverse=True
        ),
        "spearman_correlation_pagerank_vs_bridge_score": (
            float(bridge_corr)
            if pd.notna(bridge_corr) else None
        ),
        "spearman_p_value": (
            float(bridge_corr_p)
            if pd.notna(bridge_corr_p) else None
        ),
        "mean_bridge_score_top_pagerank": (
            mean(top_bridge_scores) if top_bridge_scores else None
        ),
        "mean_bridge_score_rest_of_network": (
            mean(other_bridge_scores) if other_bridge_scores else None
        ),
        "mann_whitney_u_statistic": (
            float(mw_stat) if mw_stat is not None else None
        ),
        "mann_whitney_p_value": (
            float(mw_p) if mw_p is not None else None
        ),
    }
    
summary["rq4_bridging_communities"] = rq4_results


# Write per-node CSV
# make sure the output folder exists
PAGERANK_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

with OUTPUT_NODE_CSV.open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)

    # header: the community and bridge columns come from louvain and leiden
    writer.writerow([
        "artist_id",
        "artist_name",
        "degree",
        "weighted_degree",
        "pagerank_weighted",
        "pagerank_unweighted",
        "degree_rank",
        "pagerank_rank",
        "rank_diff",
        "community_id_louvain",
        "bridge_score_louvain",
        "community_id_leiden",
        "bridge_score_leiden",
    ])

    for node in G.nodes():
        writer.writerow([
            node,
            name_of(node),
            degree_dict[node],
            weighted_degree_dict[node],
            pagerank_weighted[node],
            pagerank_unweighted[node],
            degree_rank[node],
            pagerank_rank[node],
            rank_diff[node],
            community_mappings["louvain"][node],
            bridge_scores_by_algorithm["louvain"][node],
            community_mappings["leiden"][node],
            bridge_scores_by_algorithm["leiden"][node],
        ])

# plots
# bar chart of the 15 highest weighted pagerank values
top_15 = sorted(pagerank_weighted, key=pagerank_weighted.get, reverse=True)[:15]
plt.figure(figsize=(10, 7))
plt.barh([name_of(n) for n in reversed(top_15)], [pagerank_weighted[n] for n in reversed(top_15)])
plt.xlabel("Weighted PageRank")
plt.title("Top 15 artists by PageRank")
plt.tight_layout()
plt.savefig(OUTPUT_TOP_PLOT, dpi=150)
plt.close()

# scatter of degree rank against pagerank rank: green when pagerank ranks the artist higher, red when lower
plt.figure(figsize=(8, 8))
colors = ["tab:green" if rank_diff[n] > 0 else "tab:red" for n in G.nodes()]
plt.scatter(
    [degree_rank[n] for n in G.nodes()],
    [pagerank_rank[n] for n in G.nodes()],
    c=colors, alpha=0.4, s=15,
)
plt.plot([1, G.number_of_nodes()], [1, G.number_of_nodes()], "k--", linewidth=1, label="degree rank = PageRank rank")
plt.xlabel("Degree rank (1 = most collaborators)")
plt.ylabel("PageRank rank (1 = highest PageRank)")
plt.title("PageRank rank vs. degree rank\n(green = PageRank ranks them higher, red = lower)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_SCATTER_PLOT, dpi=150)
plt.close()