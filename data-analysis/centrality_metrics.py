# centrality_metrics.py

import csv
from pathlib import Path

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "centrality_metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ARTISTS_CSV = DATA_DIR / "spotify_artists_2000_2022.csv"
COLLABS_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"

METRICS = [
    "degree_centrality",
    "betweenness_centrality",
    "closeness_centrality",
    "eigenvector_centrality",
]

def create_graph(artists_csv, collabs_csv):
    G = nx.Graph()

    with artists_csv.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            G.add_node(row["artist_id"], name=row["artist_name"])
            
    with collabs_csv.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            a1 = row["artist_1_id"]
            a2 = row["artist_2_id"]
            
            if a1 == a2:
                continue
            
            if a1 not in G:
                G.add_node(a1, name=row["artist_1_name"])
            
            if a2 not in G:
                G.add_node(a2, name=row["artist_2_name"])
            
            G.add_edge(a1, a2, weight=int(row["shared_song_count"]))
    
    return G 
   
# Centrality metrics
def calculate_centrality(G):
    return {
        "degree_centrality": nx.degree_centrality(G),
        "betweenness_centrality": nx.betweenness_centrality(G, normalized=True, weight=None),
        "closeness_centrality": nx.closeness_centrality(G),
        "eigenvector_centrality": nx.eigenvector_centrality(G, max_iter=2000, tol=1e-06),
        "weighted_degree": dict(G.degree(weight="weight"))
    }

# Create results table
def create_results_table(G, centralities):
    rows = []

    for artist_id in G.nodes():
        rows.append({
            "artist_id": artist_id,
            "artist_name": G.nodes[artist_id]["name"],
            "degree": G.degree(artist_id),
            **{
                metric: values.get(artist_id, 0)
                for metric, values in centralities.items()
            }
        })
    
    return (
        pd.DataFrame(rows).sort_values("degree_centrality", ascending=False)
    )
    

# Find top 20 artists for each centrality measures
def compare_centrality_rankings(df, output_dir):
    comparison = df.copy()
    
    # Calculate rankings for every artist
    # Higher centrality = better rank (1 is the highest)
    for metric in METRICS:
        comparison[metric + "_rank"] = (
            comparison[metric]
            .rank(ascending=False, method="min")
            .astype(int)
        )

    # Select Top 20 artists based on degree centrality
    top20 = comparison.nlargest(20, "degree_centrality").copy()
    
    # Select and rename columns
    top20 = top20[[
        "artist_id",
        "artist_name",
        "degree",
        "degree_centrality_rank",
        "betweenness_centrality_rank",
        "closeness_centrality_rank",
        "eigenvector_centrality_rank"
    ]]

    top20.columns = [
        "Artist ID",
        "Artist Name",
        "Collaborators",
        "Degree Rank",
        "Betweenness Rank",
        "Closeness Rank",
        "Eigenvector Rank"
    ]

    # Save comparison table
    top20.to_csv(
        output_dir / "top20_centrality_comparison_degree.csv",
        index=False
    )
    
    return top20

# Visualize centrality rankings
def plot_centrality_heatmap(top20, output_dir):
    heatmap_data = top20.set_index("Artist Name")[[
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
        output_dir / "centrality_top20_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



# Visualize top 20 artists for each metric
def plot_centrality_bars(df, output_dir):
    for metric in METRICS:
        top20 = df.nlargest(20, metric).sort_values(metric)
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
            output_dir / f"{metric}-top20.png",
            dpi=300,
            bbox_inches="tight"
        )
    
        plt.close()
    

# Supplementary analysis: compare degree and weighted degree
def compare_weighted_degree(df, output_dir):

    # Rank every artist by degree and weighted degree
    comparison = df[[
        "artist_id",
        "artist_name",
        "degree",
        "weighted_degree"
    ]].copy()

    comparison["degree_rank"] = (
        comparison["degree"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    comparison["weighted_degree_rank"] = (
        comparison["weighted_degree"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Select the Top 20 by weighted degree
    top20 = comparison.nlargest(
        20, "weighted_degree"
    ).copy()

    # Save the comparison table
    top20.to_csv(
        output_dir / "top20_weighted_degree_comparison.csv",
        index=False
    )
    
    return top20

# Visualize Top 20 weighted degree
def plot_weighted_degree(top20, output_dir):
    top20 = top20.sort_values("weighted_degree")

    plt.figure(figsize=(12, 10))
    plt.barh(
        top20["artist_name"],
        top20["weighted_degree"],
        color="steelblue"
    )

    plt.xlabel("Sum of Shared-Song Counts Across Collaborators")
    plt.ylabel("Artist")
    plt.title("Top 20 Artists by Weighted Degree (Node Strength)")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        output_dir / "weighted_degree-top20.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# Scatter plot: Degree vs. Node Strength
def plot_degree_vs_strength(df, output_dir):
    for log_scale in [False, True]:
        plt.figure(figsize=(10, 7))

        sns.scatterplot(
            data=df,
            x="degree",
            y="weighted_degree",
            alpha=0.35 if log_scale else 0.5,
            color="steelblue"
        )

        if log_scale:
            plt.xscale("log")
            plt.yscale("log")

        plt.xlabel("Degree (Number of Unique Collaborators)")
        plt.ylabel("Node Strength (Sum of Shared-Song Counts)")
        plt.title(f"Relationship Between Degree and Node Strength{' (Log Scale)' if log_scale else ''}")
        plt.grid(alpha=0.2)
        plt.tight_layout()

        filename = (
            "degree_vs_node_strength_log_scatter.png"
            if log_scale
            else "degree_vs_node_strength_scatter.png"
        )

        plt.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()
        
# Main function
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build full graph
    G = create_graph(ARTISTS_CSV, COLLABS_CSV)

    print(f"Total artists: {G.number_of_nodes()}")
    print(f"Total collaborations: {G.number_of_edges()}")

    # Calculate metrics
    centralities = calculate_centrality(G)
    print("Centrality calculation completed!")

    # Save full results
    centrality_df = create_results_table(G, centralities)

    centrality_df.to_csv(
        RESULTS_DIR / "centrality_metrics.csv",
        index=False
    )

    # Print top 20 for each metric
    for metric in METRICS:
        print(f"\nTop 20 artists by {metric}:")
        top20 = centrality_df.nlargest(20, metric)
        print(top20[["artist_name", metric]].to_string(index=False))

    # Compare rankings
    top20_comparison = compare_centrality_rankings(
        centrality_df, RESULTS_DIR
    )

    print("\nTop 20 centrality comparison by degree:")
    print(top20_comparison.to_string(index=False))

    # Generate centrality visualizations
    plot_centrality_heatmap(top20_comparison, RESULTS_DIR)
    plot_centrality_bars(centrality_df, RESULTS_DIR)

    # Weighted degree analysis
    top20_weighted = compare_weighted_degree(
        centrality_df, RESULTS_DIR
    )

    print("\nTop 20 artists by weighted degree:")
    print(top20_weighted.to_string(index=False))

    plot_weighted_degree(top20_weighted, RESULTS_DIR)
    plot_degree_vs_strength(centrality_df, RESULTS_DIR)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()