# centrality_metrics.py
# Script to calculate the centrality metrics of the artist network

import csv
from pathlib import Path

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

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

# Largest connected component sub

# Find largest connected component
largest_component = max(nx.connected_components(G), key=len)

# Create a separate subgraph for the largest connected component
G_main = G.subgraph(largest_component).copy()

print("Largest connected component:")
print(f"Total artists: {G_main.number_of_nodes()}")
print(f"Total collaborations: {G_main.number_of_edges()}")

# Centrality metrics for G_main (largest connected component)
#
# Centrality metrics are calculated on the
# largest connected component.
#
# All four metrics are unweighted.
# Each collaboration counts as one connection,
# regardless of the number of shared songs.


# 1. Degree centrality
degree_centrality = nx.degree_centrality(G_main)

# 2. Betweeness centrality
betweenness_centrality = nx.betweenness_centrality(G_main, normalized=True, weight=None)

# 3. Closeness centrality
closeness_centrality = nx.closeness_centrality(G_main)

# 4. Eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(
  G_main,
  max_iter=2000,
  tol=1e-06
)

print("Centrality calculation completed!")

# Create results table

results = []

for artist_id in G_main.nodes():
  results.append({
    "artist_id": artist_id,
    "artist_name": G_main.nodes[artist_id]["name"],
    "degree": G_main.degree(artist_id),
    "degree_centrality": degree_centrality.get(artist_id, 0),
    "betweenness_centrality": betweenness_centrality.get(artist_id, 0),
    "closeness_centrality": closeness_centrality.get(artist_id, 0),
    "eigenvector_centrality": eigenvector_centrality.get(artist_id, 0),
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
metrics = ["degree_centrality", "betweenness_centrality", "closeness_centrality", "eigenvector_centrality"]

for metric in metrics:
  print(f"Top 20 artists by {metric}:")
  top20 = centrality_df.nlargest(20, metric)
  print(top20[["artist_name", metric]].to_string(index=False))
  
  
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
