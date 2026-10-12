# genre_analysis.py
# Script to analyze the genre distribution and collaborations of artists

import csv
from pathlib import Path

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "genre_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ARTISTS_CSV = DATA_DIR / "spotify_artists_2000_2022.csv"
COLLABS_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"

# For lovain community detection
seed = 42

# Build graph
G = nx.Graph()

# 1. Main graph construction

# Add artists

# Add artists with genre attributes
with ARTISTS_CSV.open("r", encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):

        genre = row["main_genre"].strip()

        G.add_node(
            row["artist_id"],
            name=row["artist_name"],
            main_genre=genre if genre else None
        )
    
# Add collaborations
with COLLABS_CSV.open("r", encoding="utf-8", newline="") as file:
  for row in csv.DictReader(file):
    a1 = row["artist_1_id"]
    a2 = row["artist_2_id"]
    weight = int(row["shared_song_count"])
    
    if a1 not in G:
        G.add_node(
            a1,
            name=row["artist_1_name"],
            main_genre=None
        )

    if a2 not in G:
        G.add_node(
            a2,
            name=row["artist_2_name"],
            main_genre=None
        )
      
    G.add_edge(a1, a2, weight=weight)
  
print(f"Total artists: {G.number_of_nodes()}")
print(f"Total collaborations: {G.number_of_edges()}")

# Largest connected component sub

# Find largest connected component
largest_component = max(nx.connected_components(G), key=len)

# Create a separate subgraph for the largest connected component
G_main = G.subgraph(largest_component).copy()

known_genres_graph = sum(
    G_main.nodes[artist_id].get("main_genre") is not None
    for artist_id in G_main.nodes()
)

print(f"Known genres attached to graph: {known_genres_graph}")

print("Largest connected component:")
print(f"Total artists: {G_main.number_of_nodes()}")
print(f"Total collaborations: {G_main.number_of_edges()}")

# Genre analysis

# First find out how many artists have genres
artists_df = pd.read_csv(
    DATA_DIR / "spotify_artists_2000_2022.csv"
)

# Count available genre values
total = len(artists_df)
known = artists_df["main_genre"].notna().sum()
missing = total - known

print(f"Total artists: {total}")
print(f"Artists with genre: {known}")
print(f"Artists without genre: {missing}")
print(f"Genre coverage: {known / total * 100:.1f}%")

# Inspect the most common genres
print("\nTop 20 genres:")
print(
    artists_df["main_genre"]
    .value_counts()
    .head(20)
)

# Genre coverage within the largest connected component

genre_lookup = dict(
    zip(
        artists_df["artist_id"],
        artists_df["main_genre"]
    )
)

known_genres_lcc = sum(
    pd.notna(genre_lookup.get(artist_id))
    for artist_id in G_main.nodes()
)

print(f"Artists in LCC: {G_main.number_of_nodes()}")
print(f"Known genres in LCC: {known_genres_lcc}")
print(
    f"Genre coverage in LCC: "
    f"{known_genres_lcc / G_main.number_of_nodes() * 100:.1f}%"
)


# 2. Louvain community detection

communities = nx.community.louvain_communities(
    G_main,
    weight=None,
    resolution=1,
    seed=42
)

# Sort communities by size (largest first)
communities = sorted(communities, key=len, reverse=True)

print(f"\nNumber of communities: {len(communities)}")

modularity = nx.community.modularity(
    G_main,
    communities,
    weight=None
)

print(f"Modularity: {modularity:.4f}")

# Assign community IDs (1 = largest community)
community_mapping = {
    artist_id: community_id
    for community_id, community in enumerate(communities, start=1)
    for artist_id in community
}

nx.set_node_attributes(G_main, community_mapping, "community")


# 3. Save artist community assignments

results = []

for artist_id in G_main.nodes():
    attributes = G_main.nodes[artist_id]

    results.append({
        "artist_id": artist_id,
        "artist_name": attributes["name"],
        "main_genre": attributes.get("main_genre"),
        "community": community_mapping[artist_id],
        "degree": G_main.degree(artist_id)
    })

community_df = pd.DataFrame(results)

community_df = community_df.sort_values(
    ["community", "degree"],
    ascending=[True, False]
)

community_df.to_csv(
    RESULTS_DIR / "artist_community_assignments.csv",
    index=False
)


# 4. Community size analysis

community_sizes = (
    community_df.groupby("community")
    .size()
    .reset_index(name="size")
    .sort_values("size", ascending=False)
)

community_sizes.to_csv(
    RESULTS_DIR / "community_sizes.csv",
    index=False
)

print("\nTop 10 largest communities:")
print(community_sizes.head(10).to_string(index=False))

# Bar chart: Top 20 largest communities
top20_sizes = community_sizes.head(20).sort_values("size")

plt.figure(figsize=(11, 8))

plt.barh(
    top20_sizes["community"].astype(str),
    top20_sizes["size"],
    color="steelblue"
)

plt.xlabel("Number of Artists")
plt.ylabel("Community ID")
plt.title("Top 20 Largest Collaboration Communities")
plt.grid(axis="x", alpha=0.2)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "community_sizes_top20.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 5. Genre coverage by community

genre_coverage = (
    community_df.groupby("community")
    .agg(
        total_artists=("artist_id", "size"),
        known_genres=("main_genre", "count")
    )
    .reset_index()
)

genre_coverage["missing_genres"] = (
    genre_coverage["total_artists"]
    - genre_coverage["known_genres"]
)

genre_coverage["genre_coverage_pct"] = (
    genre_coverage["known_genres"]
    / genre_coverage["total_artists"]
    * 100
).round(1)

genre_coverage.to_csv(
    RESULTS_DIR / "community_genre_coverage.csv",
    index=False
)

print("\nGenre coverage of the 10 largest communities:")
print(genre_coverage.head(10).to_string(index=False))


# 6. Genre composition by community

# Only artists with known genres
known_genres_df = community_df.dropna(
    subset=["main_genre"]
).copy()

# Count artists per genre in each community
genre_counts = (
    known_genres_df.groupby(["community", "main_genre"])
    .size()
    .reset_index(name="artist_count")
)

genre_counts.to_csv(
    RESULTS_DIR / "community_genre_counts.csv",
    index=False
)

# Find most common genre in each community
dominant_genres = (
    genre_counts.sort_values(
        ["community", "artist_count"],
        ascending=[True, False]
    )
    .drop_duplicates(subset=["community"])
    .rename(columns={
        "main_genre": "most_common_genre",
        "artist_count": "genre_count"
    })
)

community_summary = genre_coverage.merge(
    dominant_genres[[
        "community",
        "most_common_genre",
        "genre_count"
    ]],
    on="community",
    how="left"
)

community_summary["dominant_genre_pct"] = (
    community_summary["genre_count"]
    / community_summary["known_genres"]
    * 100
).round(1)

community_summary.to_csv(
    RESULTS_DIR / "community_genre_summary.csv",
    index=False
)

print("\nCommunity genre summary:")
print(community_summary.head(10).to_string(index=False))


# 7. Genre composition heatmap

# Focus on the 10 largest communities
top_communities = community_sizes.head(10)["community"]

# Focus on the 15 most common known genres overall
top_genres = (
    known_genres_df["main_genre"]
    .value_counts()
    .head(15)
    .index
)

heatmap_df = known_genres_df[
    known_genres_df["community"].isin(top_communities)
    & known_genres_df["main_genre"].isin(top_genres)
]

genre_matrix = pd.crosstab(
    heatmap_df["community"],
    heatmap_df["main_genre"]
)

genre_matrix = genre_matrix.reindex(
    index=top_communities,
    columns=top_genres,
    fill_value=0
)

plt.figure(figsize=(15, 8))

print("Known genre artists:", len(known_genres_df))
print("Filtered heatmap artists:", len(heatmap_df))
print("Genre matrix shape:", genre_matrix.shape)
print("Top communities:", top_communities.tolist())
print("Top genres:", top_genres.tolist())

sns.heatmap(
    genre_matrix,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={"label": "Number of Artists"}
)

plt.title("Genre Composition of the 10 Largest Communities")
plt.xlabel("Main Genre")
plt.ylabel("Community ID")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "community_genre_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nCommunity analysis completed!")
