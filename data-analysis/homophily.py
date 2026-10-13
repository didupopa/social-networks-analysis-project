import argparse
import ast
import networkx as nx
import pandas as pd

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

    G = create_graph(artists, collaborations)
    edges = list(G.edges)
    overall = stats_for_edges(edges, markets)
    overall['market_labelled_artists'] = sum(bool(markets[n]) for n in G)
    pd.DataFrame([overall]).to_csv(args.output / 'market_homophily.csv', index=False)

    rows = []
    for (algorithm, community_id), data in assignments.groupby(["algorithm", "community_id"]):
        members = set(data.artist_id)
        internal_edges = [(u, v) for u, v in G.subgraph(members).edges]
        result = stats_for_edges(internal_edges, markets)
        rows.append({
            "algorithm": algorithm,
            "community_id": community_id,
            "size": len(members),
            "market_labelled_artists": sum(bool(markets[n]) for n in members),
            **result,
        })
    pd.DataFrame(rows).to_csv(args.output / 'community_homophily.csv', index=False)
    print(f"results saved to {args.output}")


if __name__ == "__main__":
    main()        