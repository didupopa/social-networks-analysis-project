# centrality_metrics.py
# Script to calculate the centrality metrics of the artist network

import csv
from pathlib import Path

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "centrality_metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ARTISTS_CSV = DATA_DIR / "spotify_artists_2000_2022.csv"
COLLABS_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"

# Build graph
G = nx.Graph()

# Main graph construction

# Add artists
with ARTISTS_CSV.open("r", encoding="utf-8", newline="") as file:
  for row in csv.DictReader(file):
    G.add_node(row["artist_id"], name=row["artist_name"])
    
# Add collaborations
with COLLABS_CSV.open("r", encoding="utf-8", newline="") as file:
  for row in csv.DictReader(file):
    a1 = row["artist_1_id"]
    a2 = row["artist_2_id"]
    weight = int(row["shared_song_count"])
    
    if a1 not in G:
      G.add_node(a1, name=row["artist_1_name"])
      
    if a2 not in G:
      G.add_node(a2, name=row["artist_2_name"])
      
    G.add_edge(a1, a2, weight=weight)
  
print(f"Total artists: {G.number_of_nodes()}")
print(f"Total collaborations: {G.number_of_edges()}")

# Centrality metrics
#
# All four metrics are unweighted.
# Each collaboration counts as one connection,
# regardless of the number of shared songs.


# 1. Degree centrality
degree_centrality = nx.degree_centrality(G)

# 2. Betweeness centrality
betweenness_centrality = nx.betweenness_centrality(G, normalized=True, weight=None)

# 3. Closeness centrality
closeness_centrality = nx.closeness_centrality(G)

# 4. Eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(
  G,
  max_iter=2000,
  tol=1e-06
)

# 5. Supplementary analysis: weighted degree (node strength)
# Sum of shared songs across an artist's collaborations
weighted_degree = dict(G.degree(weight="weight"))


print("Centrality calculation completed!")

# Create results table

results = []

for artist_id in G.nodes():
  results.append({
    "artist_id": artist_id,
    "artist_name": G.nodes[artist_id]["name"],
    "degree": G.degree(artist_id),
    "degree_centrality": degree_centrality.get(artist_id, 0),
    "betweenness_centrality": betweenness_centrality.get(artist_id, 0),
    "closeness_centrality": closeness_centrality.get(artist_id, 0),
    "eigenvector_centrality": eigenvector_centrality.get(artist_id, 0),
    "weighted_degree": weighted_degree.get(artist_id, 0),
  })
  
centrality_df = pd.DataFrame(results)

# Sort by degree centrality
centrality_df = centrality_df.sort_values(
  by="degree_centrality",
  ascending=False
)

# Save result
centrality_df.to_csv(
  RESULTS_DIR / "centrality_metrics.csv",
  index=False
)

# Find top 20 artists for each centrality measures
metrics = [
  "degree_centrality", 
  "betweenness_centrality", 
  "closeness_centrality", 
  "eigenvector_centrality"
]

# Print top 20 artists for each centrality metric
for metric in metrics:
  print(f"Top 20 artists by {metric}:")
  top20 = centrality_df.nlargest(20, metric)
  print(top20[["artist_name", metric]].to_string(index=False))
  
# Comparison of centrality metrics

comparison_df = centrality_df.copy()

# Calculate rankings for every artist
# Higher centrality = better rank (1 is the highest)
for metric in metrics:
    comparison_df[metric + "_rank"] = (
        comparison_df[metric]
        .rank(ascending=False, method="min")
        .astype(int)
    )

# Select Top 20 artists based on degree centrality
top20_comparison = comparison_df.nlargest(
    20,
    "degree_centrality"
).copy()

# Select and rename columns
top20_comparison = top20_comparison[[
    "artist_id",
    "artist_name",
    "degree",
    "degree_centrality_rank",
    "betweenness_centrality_rank",
    "closeness_centrality_rank",
    "eigenvector_centrality_rank"
]]

top20_comparison.columns = [
    "Artist ID",
    "Artist Name",
    "Collaborators",
    "Degree Rank",
    "Betweenness Rank",
    "Closeness Rank",
    "Eigenvector Rank"
]

# Save comparison table
top20_comparison.to_csv(
    RESULTS_DIR / "top20_centrality_comparison_degree.csv",
    index=False
)

print("\nTop 20 centrality comparison by degree:")
print(top20_comparison.to_string(index=False))

# Visualise comparison with heatmap
heatmap_data = top20_comparison.set_index("Artist Name")

# Select only numerical ranking columns
heatmap_data = heatmap_data[[
    "Degree Rank",
    "Betweenness Rank",
    "Closeness Rank",
    "Eigenvector Rank"
]]

plt.figure(figsize=(10, 12))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt="d",
    cmap="YlGnBu_r",
    linewidths=0.5,
    cbar_kws={"label": "Centrality Rank (Lower is Better)"}
)

plt.title("Centrality Rank Comparison of the 20 Most Connected Artists")
plt.xlabel("Centrality Metric")
plt.ylabel("Artist")
plt.xticks(rotation=30, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "centrality_top20_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Visualize centrality rankings

# Bar chart for each metric
for metric in metrics:
   top20 = centrality_df.nlargest(20, metric)
   top20 = top20.sort_values(metric)
   
   metric_name = metric.replace("_", " ").title()
   
   plt.figure(figsize=(12, 10))
   
   plt.barh(
     top20["artist_name"],
     top20[metric],
     color="steelblue"
   )
     
   plt.xlabel(metric_name)
   plt.ylabel("Artist")
   plt.title(f"Top 20 Artists by {metric_name}")
   
   plt.grid(axis="x", alpha=0.2)
   plt.tight_layout()
   
   plt.savefig(
     RESULTS_DIR / f"{metric}-top20.png",
     dpi=300,
     bbox_inches="tight"
   )
   
   plt.close()
   

# Supplementary analysis: weighted degree

# Rank every artist by degree and weighted degree
weighted_comparison = centrality_df[[
    "artist_id",
    "artist_name",
    "degree",
    "weighted_degree"
]].copy()

weighted_comparison["degree_rank"] = (
    weighted_comparison["degree"]
    .rank(ascending=False, method="min")
    .astype(int)
)

weighted_comparison["weighted_degree_rank"] = (
    weighted_comparison["weighted_degree"]
    .rank(ascending=False, method="min")
    .astype(int)
)

# Select the Top 20 by weighted degree
top20_weighted = weighted_comparison.nlargest(
    20, "weighted_degree"
).copy()

# Save the comparison table
top20_weighted.to_csv(
    RESULTS_DIR / "top20_weighted_degree_comparison.csv",
    index=False
)

print("\nTop 20 artists by weighted degree:")
print(top20_weighted.to_string(index=False))

# Visualize Top 20 weighted degree
top20_plot = top20_weighted.sort_values("weighted_degree")

plt.figure(figsize=(12, 10))

plt.barh(
    top20_plot["artist_name"],
    top20_plot["weighted_degree"],
    color="steelblue"
)

plt.xlabel("Sum of Shared-Song Counts Across Collaborators")
plt.ylabel("Artist")
plt.title("Top 20 Artists by Weighted Degree (Node Strength)")
plt.grid(axis="x", alpha=0.2)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "weighted_degree-top20.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Scatter plot: Degree vs. Node Strength

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=centrality_df,
    x="degree",
    y="weighted_degree",
    alpha=0.5,
    color="steelblue"
)

plt.xlabel("Degree (Number of Unique Collaborators)")
plt.ylabel("Node Strength (Sum of Shared-Song Counts)")
plt.title("Relationship Between Degree and Node Strength")

plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "degree_vs_node_strength_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Scatter plot: Degree vs. Node Strength (log scale)

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=centrality_df,
    x="degree",
    y="weighted_degree",
    alpha=0.35,
    s=25,
    color="steelblue"
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Degree (Number of Unique Collaborators, log scale)")
plt.ylabel("Node Strength (Sum of Shared-Song Counts, log scale)")
plt.title("Relationship Between Degree and Node Strength")

plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "degree_vs_node_strength_log_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()