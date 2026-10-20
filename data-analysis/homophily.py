import argparse
import ast
import networkx as nx
import pandas as pd
import random

from pathlib import Path

# paths
ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIST_CSV = DATA_DIR / "spotify_artists_2000_2022.csv"
COLLAB_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "homophily"
PARTITION_RESULTS = ROOT_DIR / "results" / "partitioning-algorithms" / "community_assignments.csv"


# helper function to create a graph from artists and collaborations data
def create_graph(artists, collaborations):
    G = nx.Graph()
    for artist in artists.itertuples(index=False):
        genre = str(artist.main_genre) if pd.notna(artist.main_genre) else "unknown"
        G.add_node(artist.artist_id, name=str(artist.artist_name), genre=genre)
    for collab in collaborations.itertuples(index=False):
        if collab.artist_1_id == collab.artist_2_id:
            continue
        G.add_edge(collab.artist_1_id, collab.artist_2_id, weight=int(collab.shared_song_count), first_year=int(collab.first_collab_year), last_year=int(collab.last_collab_year))

    return G


# helper function to parse the available markets column in the artist CSV
def parse_markets(value):
    if pd.isna(value) or not str(value).strip():
        return set()
    try:
        parsed = ast.literal_eval(str(value))
        return set(parsed) if isinstance(parsed, list) else set()
    except (SyntaxError, ValueError):
        return set()


# helper function to parse the main genre column in the artist CSV
def parse_genre(value):
    if pd.notna(value) and str(value).strip():
        return str(value).strip()
    return "unknown"

# helper function to compute the Jaccard similarity between two sets
def jaccard(left, right):
    return len(left & right) / len(left | right)


# helper function to compute statistics for a given set of edges and the corresponding markets
def stats_for_edges(edges, markets):
    eligible_market = [(u, v) for u, v in edges if markets.get(u) and markets.get(v)]
    return {
        "edges_total": len(edges),
        "market_eligible_edges": len(eligible_market),
        "mean_market_jaccard": (sum(jaccard(markets[u], markets[v]) for u, v in eligible_market) / len(eligible_market) if eligible_market else float("nan")),
    }
    
    
# helper function to compute statistics for a given set of edges and the corresponding genres
def genre_stats_for_edges(edges, genres):
    eligible_genres = [
        (u, v)
        for u, v in edges
        if genres.get(u) and genres.get(v)
        and genres[u] != "unknown" 
        and genres[v] != "unknown"
    ]

    same_genre = sum(genres[u] == genres[v] for u, v in eligible_genres)
    cross_genre = len(eligible_genres) - same_genre
    
    return {
        "edges_total": len(edges),
        "genre_eligible_edges": len(eligible_genres),
        "same_genre_edges": same_genre,
        "cross_genre_edges": cross_genre,
        "same_genre_pct": (
            100 * same_genre / len(eligible_genres) if eligible_genres else float("nan")
        )
    }


def us_market_homophily(G, markets, assignments, output, n_permutations=2000, seed=42):
    known = {n: markets[n] for n in G if markets.get(n)}
    us_available = {n: ('US' in value) for n, value in known.items()}
    eligible = [(u, v) for u, v in G.edges() if u in known and v in known]

    def mixing_counts(edge_list, labels):
        counts = {'US_US': 0, 'US_nonUS': 0, 'nonUS_nonUS': 0}
        for u, v in edge_list:
            a, b = labels[u], labels[v]
            if a and b:
                counts['US_US'] += 1
            elif a or b:
                counts['US_nonUS'] += 1
            else:
                counts['nonUS_nonUS'] += 1
        return counts

    counts = mixing_counts(eligible, us_available)
    total = len(eligible)
    observed_same = (counts['US_US'] + counts['nonUS_nonUS']) / total if total else float("nan")

    rng = random.Random(seed)
    nodes = sorted(known)
    labels = [us_available[n] for n in nodes]
    null_scores = []
    if total and len(set(labels)) > 1:
        for _ in range(n_permutations):
            permuted = labels.copy()
            rng.shuffle(permuted)
            randomized = dict(zip(nodes, permuted))
            randomized_counts = mixing_counts(eligible, randomized)
            null_scores.append((randomized_counts['US_US'] + randomized_counts['nonUS_nonUS']) / total)

    expected = sum(null_scores) / len(null_scores) if null_scores else float("nan")
    p_value = (1 + sum(value >= observed_same for value in null_scores)) / (len(null_scores) + 1) if null_scores else float("nan")

    summary = {
        "total_artists": G.number_of_nodes(),
        "artists_with_market_info": len(known),
        "artists_available_in_US": sum(labels),
        "artists_not_available_in_US": len(known) - sum(labels),
        "total_collaboration_edges": G.number_of_edges(),
        "eligible_edges": total,
        **counts,
        "same_US_status_rate": observed_same,
        "null_mean_same_status_rate": expected,
        "difference_from_null": observed_same - expected if null_scores else float("nan"),
        "one_sided_p_value": p_value,
        "n_permutations": len(null_scores),
    }
    pd.DataFrame([summary]).to_csv(output / "us_market_homophily.csv", index=False)

    records = []
    for (algorithm, community_id), data in assignments.groupby(["algorithm", "community_id"]):
        members = set(data.artist_id) & set(G)
        scoped = [(u, v) for u, v in G.subgraph(members).edges() if u in known and v in known]
        group_counts = mixing_counts(scoped, us_available)
        group_total = len(scoped)
        records.append({
            'algorithm': algorithm,
            'community_id': community_id,
            'community_size': len(members),
            'artists_with_market_data': sum(n in known for n in members),
            'artists_available_in_US': sum(us_available.get(n, False) for n in members),
            'artists_not_available_in_US': sum(n in known and not us_available[n] for n in members),
            'eligible_internal_edges': group_total,
            **group_counts,
            'same_US_status_rate': (
                (group_counts['US_US'] + group_counts['nonUS_nonUS']) / group_total
                if group_total else float('nan')
            ),
        })
    pd.DataFrame(records).to_csv(output / 'us_community_homophily.csv', index=False)
    print('US market homophily:', summary)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artists", type=Path, default=ARTIST_CSV)
    parser.add_argument("--collaborations", type=Path, default=COLLAB_CSV)
    parser.add_argument("--assignments", type=Path, default=PARTITION_RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    artists = pd.read_csv(args.artists, dtype={"artist_id": str})
    collaborations = pd.read_csv(args.collaborations, dtype={"artist_1_id": str, "artist_2_id": str})
    assignments = pd.read_csv(args.assignments, dtype={"artist_id": str, "algorithm": str})
    markets = {r.artist_id: parse_markets(r.available_markets) for r in artists.itertuples(index=False)}
    genres = {r.artist_id: parse_genre(r.main_genre) for r in artists.itertuples(index=False)}
    
    G = create_graph(artists, collaborations)
    edges = list(G.edges)
    overall = stats_for_edges(edges, markets)
    overall['market_labelled_artists'] = sum(bool(markets[n]) for n in G)
    pd.DataFrame([overall]).to_csv(args.output / 'market_homophily.csv', index=False)
    
    overall_genre = genre_stats_for_edges(edges, genres)
    overall_genre["genre_labelled_artists"] = sum(
        genres.get(n, "unknown") != "unknown" 
        for n in G
    )
    overall_genre["edges_total"] = len(edges)
    pd.DataFrame([overall_genre]).to_csv(args.output / "genre_homophily.csv", index=False)

    rows = []
    for (algorithm, community_id), data in assignments.groupby(["algorithm", "community_id"]):
        members = set(data.artist_id)
        internal_edges = [(u, v) for u, v in G.subgraph(members).edges]
        market_result = stats_for_edges(internal_edges, markets)
        genre_result = genre_stats_for_edges(internal_edges, genres)
        rows.append({
            "algorithm": algorithm,
            "community_id": community_id,
            "size": len(members),
            "market_labelled_artists": sum(bool(markets[n]) for n in members),
            "genre_labelled_artists": sum(
                genres.get(n, "unknown") != "unknown" 
                for n in members
            ),
            **market_result,
            **genre_result,
        })
    pd.DataFrame(rows).to_csv(args.output / 'community_homophily.csv', index=False)
    us_market_homophily(G, markets, assignments, args.output)
    print(f"results saved to {args.output}")


if __name__ == "__main__":
    main()        