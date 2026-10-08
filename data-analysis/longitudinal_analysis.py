import csv
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "longitudinal"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COLLABS_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"

OUTPUT_LONG_CSV = RESULTS_DIR / "longitudinal-clustering-long.csv"
OUTPUT_COEFF_CSV = RESULTS_DIR / "longitudinal-clustering-by-year.csv"
OUTPUT_RANK_CSV = RESULTS_DIR / "longitudinal-clustering-ranks-by-year.csv"
OUTPUT_SUMMARY_CSV = RESULTS_DIR / "longitudinal-clustering-summary.csv"
OUTPUT_LINE_PLOT = RESULTS_DIR / "longitudinal-clustering-lines.png"
OUTPUT_RANK_PLOT = RESULTS_DIR / "longitudinal-clustering-rank-heatmap.png"

TOP_N = 20

# The network for year Y contains every collaboration with a shared song released
# in the last WINDOW_YEARS years: [Y - WINDOW_YEARS + 1, Y].
WINDOW_YEARS = 1

# Clustering needs at least 1 neighbour to be defined. Artists with fewer than
# this many collaborators in a given year's network get no rank for that year.
MIN_DEGREE_FOR_RANKING = 1


#load collaborations

edges = []  # (artist_1_id, artist_2_id, set of years with a shared song)
names = {}

with COLLABS_CSV.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        a1, a2 = row["artist_1_id"], row["artist_2_id"]
        names[a1] = row["artist_1_name"]
        names[a2] = row["artist_2_name"]

        collab_years = {int(y) for y in row["collab_years"].split(";") if y}

        if collab_years:
            edges.append((a1, a2, collab_years))

print(f"Loaded {len(edges)} collaboration edges (window = {WINDOW_YEARS} year(s))")

# step 1: top artists by degree centrality in the FULL graph
# (degree centrality = degree / (n - 1), so ranking by degree is identical)

G_full = nx.Graph()
G_full.add_edges_from((a1, a2) for a1, a2, _ in edges)

degree_centrality = nx.degree_centrality(G_full)

top_artists = sorted(
    degree_centrality,
    key=lambda n: (-degree_centrality[n], names.get(n, "")),
)[:TOP_N]

print(f"\nTop {TOP_N} artists by degree centrality (full graph):")
for i, node in enumerate(top_artists, start=1):
    print(f"  {i:2d}. {names[node]} (degree {G_full.degree(node)}, "
          f"centrality {degree_centrality[node]:.4f})")

# step 2: yearly graphs and clustering coefficients

all_years = [y for _, _, ys in edges for y in ys]
years = list(range(min(all_years), max(all_years) + 1))

coefficient = {}     # (artist, year) -> clustering coefficient
degree_in_year = {}  # (artist, year) -> degree in that year's network

for year in years:
    window = set(range(year - WINDOW_YEARS + 1, year + 1))

    G_year = nx.Graph()
    G_year.add_edges_from((a, b) for a, b, ys in edges if ys & window)

    # artists absent from this year's network have no collaborations in the window
    present = [n for n in top_artists if n in G_year]
    clustering = nx.clustering(G_year, nodes=present) if present else {}

    for node in top_artists:
        degree_in_year[(node, year)] = G_year.degree(node) if node in G_year else 0
        coefficient[(node, year)] = clustering.get(node, 0.0)

# step 3: rank the top artists within each year

rank = {}  # (artist, year) -> rank among rankable artists, or None

for year in years:
    rankable = [n for n in top_artists if degree_in_year[(n, year)] >= MIN_DEGREE_FOR_RANKING]
    ordered = sorted(rankable, key=lambda n: (-coefficient[(n, year)], names[n]))

    previous_value = None
    previous_rank = 0

    for position, node in enumerate(ordered, start=1):
        value = coefficient[(node, year)]
        # tied coefficients share a rank
        current_rank = previous_rank if value == previous_value else position
        rank[(node, year)] = current_rank
        previous_value, previous_rank = value, current_rank

    for node in top_artists:
        rank.setdefault((node, year), None)

# write CSVs

with OUTPUT_LONG_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["year", "artist_id", "artist_name", "degree_in_year",
                     "clustering_coefficient", "rank_in_year"])
    for year in years:
        for node in top_artists:
            writer.writerow([
                year, node, names[node], degree_in_year[(node, year)],
                coefficient[(node, year)],
                rank[(node, year)] if rank[(node, year)] is not None else "",
            ])

with OUTPUT_COEFF_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["artist_id", "artist_name"] + years)
    for node in top_artists:
        writer.writerow([node, names[node]] + [coefficient[(node, y)] for y in years])

with OUTPUT_RANK_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["artist_id", "artist_name"] + years)
    for node in top_artists:
        writer.writerow([node, names[node]] + [
            rank[(node, y)] if rank[(node, y)] is not None else "" for y in years
        ])

# overall ranking: mean coefficient across the years the artist was rankable
summary_rows = []

for node in top_artists:
    rankable_years = [y for y in years if rank[(node, y)] is not None]
    coeffs = [coefficient[(node, y)] for y in rankable_years]
    ranks = [rank[(node, y)] for y in rankable_years]

    summary_rows.append({
        "artist_id": node,
        "artist_name": names[node],
        "final_degree": G_full.degree(node),
        "years_rankable": len(rankable_years),
        "mean_clustering": sum(coeffs) / len(coeffs) if coeffs else None,
        "mean_rank": sum(ranks) / len(ranks) if ranks else None,
        "best_rank": min(ranks) if ranks else None,
        "final_year_clustering": coefficient[(node, years[-1])],
    })

summary_rows.sort(key=lambda r: (r["mean_clustering"] is None, -(r["mean_clustering"] or 0)))

with OUTPUT_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

# plots

window_label = "that year only" if WINDOW_YEARS == 1 else f"{WINDOW_YEARS}-year window"

plt.figure(figsize=(14, 8))
for node in top_artists:
    plt.plot(years, [coefficient[(node, y)] for y in years], marker="o", markersize=3, label=names[node])
plt.xlabel("Year")
plt.ylabel("Clustering coefficient")
plt.title(f"Clustering coefficient of the top {TOP_N} artists by degree ({window_label})")
plt.legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.savefig(OUTPUT_LINE_PLOT, dpi=150)
plt.close()

# heatmap of in-year ranks (1 = highest clustering that year); blank = not rankable
order_for_plot = [r["artist_id"] for r in summary_rows]
grid = [
    [rank[(node, y)] if rank[(node, y)] is not None else float("nan") for y in years]
    for node in order_for_plot
]

plt.figure(figsize=(14, 8))
image = plt.imshow(grid, aspect="auto", cmap="viridis_r")
plt.colorbar(image, label="Rank in year (1 = highest clustering)")
plt.xticks(range(len(years)), years, rotation=90, fontsize=8)
plt.yticks(range(len(order_for_plot)), [names[n] for n in order_for_plot], fontsize=8)
plt.title(f"Yearly clustering rank among the top artists ({window_label}; blank = no collaborations)")
plt.tight_layout()
plt.savefig(OUTPUT_RANK_PLOT, dpi=150)
plt.close()