import csv
import json
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
NETWORK_ANALYSIS_DIR = RESULTS_DIR / "network-analysis"
ARTISTS_CSV = DATA_DIR / "spotify_artists_2000_2022.csv"
COLLABS_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"

# output files in the network-analysis results folder
OUTPUT_SUMMARY_JSON = RESULTS_DIR / NETWORK_ANALYSIS_DIR / "network-statistics-summary.json"
OUTPUT_NODE_METRICS_CSV = RESULTS_DIR / NETWORK_ANALYSIS_DIR / "network-statistics-per-node.csv"
OUTPUT_NETWORK_GRAPH_PNG = RESULTS_DIR / NETWORK_ANALYSIS_DIR / "network-graph.png"

# network graph visualization options
VISUALIZE_ONLY_LARGEST_COMPONENT = False   # False = plot the whole graph; True = only the connected part
VISUALIZE_LABEL_TOP_N = 25                # how many highest-degree artists to label; 0 = no labels
VISUALIZE_FIGURE_SIZE = (24, 24)          # inches;
VISUALIZE_DPI = 150


# build the graph
G = nx.Graph()  # undirected: a collaboration has no inherent direction

# add every artist from the artists CSV as a node 
with ARTISTS_CSV.open("r", newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        G.add_node(row["artist_id"], name=row["artist_name"])

# add edges from the collaborations CSV, weighted by shared song count
with COLLABS_CSV.open("r", newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        a1, a2 = row["artist_1_id"], row["artist_2_id"]
        weight = int(row["shared_song_count"])

        # make sure both endpoints exist as nodes
        if a1 not in G:
            G.add_node(a1, name=row["artist_1_name"])
        if a2 not in G:
            G.add_node(a2, name=row["artist_2_name"])

        # the weight of an edge is the number of shared songs
        G.add_edge(a1, a2, weight=weight)

print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# basic counts
num_vertices = G.number_of_nodes()
num_edges = G.number_of_edges()
density = nx.density(G)

# degree distribution
# undirected graph: in-degree = out-degree = degree for every node
degree_dict = dict(G.degree())
degree_values = list(degree_dict.values())

# maps a degree to the number of artists with that degree
degree_distribution = {}
for degree in degree_values:
    degree_distribution[degree] = degree_distribution.get(degree, 0) + 1

# clustering coefficient
# clustering per artist and the average over all artists (artists with fewer than 2 neighbours count as 0)
clustering_per_node = nx.clustering(G)
average_clustering = nx.average_clustering(G)

# connected components
components = list(nx.connected_components(G))
# component sizes, largest first
component_sizes = sorted((len(c) for c in components), reverse=True)
num_components = len(components)

# diameter 
# diameter is only defined for a connected graph, so it is computed on the largest component
largest_component_nodes = max(components, key=len)
largest_subgraph = G.subgraph(largest_component_nodes)

# a single node has diameter 0
if largest_subgraph.number_of_nodes() > 1:
    diameter_largest_component = nx.diameter(largest_subgraph)
else:
    diameter_largest_component = 0

# write the per artist metrics
# write per-node metrics CSV
with OUTPUT_NODE_METRICS_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "artist_id", "artist_name", "degree", "clustering_coefficient",
    ])

    for node in G.nodes():
        writer.writerow([
            node,
            G.nodes[node].get("name", ""),
            degree_dict[node],
            clustering_per_node[node],
        ])

# write summary JSON
summary = {
    "number_of_vertices": num_vertices,
    "number_of_edges": num_edges,
    "density": density,
    "degree_distribution": degree_distribution,
    "average_clustering_coefficient": average_clustering,
    "number_of_connected_components": num_components,
    "connected_component_sizes": component_sizes,
    "diameter_of_largest_connected_component": diameter_largest_component,
    "note": (
        "This is an undirected graph (collaborations have no direction), "
        "so in-degree and out-degree are both equal to the 'degree' value "
        "for every node. Diameter is only computed on the largest connected "
        "component."
    ),
}

with OUTPUT_SUMMARY_JSON.open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)

# visualize the network as a PNG
graph_to_plot = largest_subgraph if VISUALIZE_ONLY_LARGEST_COMPONENT else G

print(f"Plotting graph: {graph_to_plot.number_of_nodes()} nodes, "
      f"{graph_to_plot.number_of_edges()} edges")

# force directed layout with a fixed seed so the picture is reproducible
pos = nx.spring_layout(graph_to_plot, k=0.15, iterations=50, seed=42)

# node size and colour both grow with degree
plot_degrees = dict(graph_to_plot.degree())
node_sizes = [20 + plot_degrees[node] * 4 for node in graph_to_plot.nodes()]
node_colors = [plot_degrees[node] for node in graph_to_plot.nodes()]

plt.figure(figsize=VISUALIZE_FIGURE_SIZE)

# edges are drawn first and faint, so the nodes end up on top
nx.draw_networkx_edges(graph_to_plot, pos, alpha=0.15, width=0.5)

nodes_drawn = nx.draw_networkx_nodes(
    graph_to_plot, pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap=plt.cm.viridis,
    linewidths=0,
)
plt.colorbar(nodes_drawn, label="Degree", shrink=0.6)

# label only the highest degree artists to keep the figure readable
if VISUALIZE_LABEL_TOP_N > 0:
    top_nodes = sorted(plot_degrees, key=plot_degrees.get, reverse=True)[:VISUALIZE_LABEL_TOP_N]
    labels = {node: graph_to_plot.nodes[node].get("name", node) for node in top_nodes}
    nx.draw_networkx_labels(graph_to_plot, pos, labels=labels, font_size=9, font_weight="bold")

plt.title(
    f"Artist collaboration network "
    f"({graph_to_plot.number_of_nodes()} nodes, {graph_to_plot.number_of_edges()} edges)"
)
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_NETWORK_GRAPH_PNG, dpi=VISUALIZE_DPI)
plt.close()