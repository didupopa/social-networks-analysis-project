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
# Centralities are unweighted except for the weighted degree, 
# because edge weight represents number of shared songs instead of distance. 
#
# NetworkX weighted betweeness and closeness centrality consider edge weights as distances,
# which is not suitable for our context where higher weights indicate stronger connections rather than longer distances.

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

# 5. Weighted degree
weighted_degree = dict(G_main.degree(weight="weight"))

print("Centrality calculation completed!")

# Create results table

results = []

for artist_id in G_main.nodes():
  results.append({
    "artist_id": artist_id,
    "artist_name": G_main.nodes[artist_id]["name"],
    "degree_centrality": degree_centrality.get(artist_id, 0),
    "betweenness_centrality": betweenness_centrality.get(artist_id, 0),
    "closeness_centrality": closeness_centrality.get(artist_id, 0),
    "eigenvector_centrality": eigenvector_centrality.get(artist_id, 0),
    "weighted_degree": weighted_degree.get(artist_id, 0)
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

print("Top 10 artists by degree centrality:")
print(centrality_df.head(10).to_string(index=False))