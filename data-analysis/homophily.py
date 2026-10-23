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
    # empty graph
    G = nx.Graph()
    for artist in artists.itertuples(index=False):
        # genre is not an entry for each artist, so we use "unknown" for missing values
        genre = str(artist.main_genre) if pd.notna(artist.main_genre) else "unknown"
        # add artist nodes
        G.add_node(artist.artist_id, name=str(artist.artist_name), genre=genre)
    for collab in collaborations.itertuples(index=False):
        # only add edges for collaborations between different artists
        if collab.artist_1_id == collab.artist_2_id:
            continue
        G.add_edge(collab.artist_1_id, collab.artist_2_id, weight=int(collab.shared_song_count), first_year=int(collab.first_collab_year), last_year=int(collab.last_collab_year))

    return G


# helper function to parse the available markets column in the artist CSV
def parse_markets(value):
    # check if market value is missing or empty
    if pd.isna(value) or not str(value).strip():
        return set()
    # check if parsed value is a list of strings
    # market entries should be like this: "["US", "GB", "CA"]"
    try:
        parsed = ast.literal_eval(str(value))
        return set(parsed) if isinstance(parsed, list) else set()
    except (SyntaxError, ValueError):
        return set()


# helper function to parse the main genre column in the artist CSV
def parse_genre(value):
    # check if genre value is not missing or not empty
    if pd.notna(value) and str(value).strip():
        return str(value).strip()
    return "unknown"

# helper function to compute the Jaccard similarity coefficient between two sets
# formula: J(A, B) = |A ∩ B| / |A ∪ B|
# similarity ranges from 0 (no overlap) to 1 (identical sets)
def jaccard(left, right):
    return len(left & right) / len(left | right)


# helper function to measure the average market similarity between pairs of collaborating artists
def stats_for_edges(edges, markets):
    # filter collaboration edges to include only those where both artists have available market info
    eligible_market = [(u, v) for u, v in edges if markets.get(u) and markets.get(v)]
    return {
        "edges_total": len(edges),                                                                                                                              # total number of collaboration edges including those without complete market information
        "market_eligible_edges": len(eligible_market),                                                                                                          # counts edges where both artists have valid market info
        "mean_market_jaccard": (sum(jaccard(markets[u], markets[v]) for u, v in eligible_market) / len(eligible_market) if eligible_market else float("nan")),  # calculat Jaccard similarity for each eligible collaborating pair and adds the scores together
    }
    
    
# helper function to compute statistics for a given set of edges and the corresponding genres
def genre_stats_for_edges(edges, genres):
    # eligible collabs are if both artists have genre info and neither genre is unknown
    eligible_genres = [(u, v) for u, v in edges if genres.get(u) and genres.get(v) and genres[u] != "unknown" and genres[v] != "unknown"]

    # count how many of the eligible edges are same-genre vs cross-genre
    same_genre = sum(genres[u] == genres[v] for u, v in eligible_genres)
    cross_genre = len(eligible_genres) - same_genre
    
    return {
        "edges_total": len(edges),                                                                              # total collab edges
        "genre_eligible_edges": len(eligible_genres),                                                           # egdes where both artists have known genres
        "same_genre_edges": same_genre,                                                                         # collabs between artists with the same genre
        "cross_genre_edges": cross_genre,                                                                       # collabs between artists with different genres
        "same_genre_pct": (100 * same_genre / len(eligible_genres) if eligible_genres else float("nan")),       # percentage of eligible collabs with matching genres
    }


# helper function to measure collaborations between artists with the same US market availability status
# for the entire network and communities detected by the partitioning algorithms (partitioning.py)
def us_market_homophily(G, markets, assignments, output):
    # dictionary containing only artists with non-empty market sets
    known = {n: markets[n] for n in G if markets.get(n)}
    # filter whether US appears in artists' available markets
    us_available = {n: ('US' in value) for n, value in known.items()}
    # collab edges where both artists have market info
    eligible = [(u, v) for u, v in G.edges() if u in known and v in known]

    # helper function to clasasify collabs according to USD availability status of the 2 artists
    def mixing_counts(edge_list, labels):
        # 3 counters: bothg artists have music available in the US,
        # only one artist has music available in the US, neither artist has music available in the US
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

    # count collabs in each US availability category using original labels
    counts = mixing_counts(eligible, us_available)
    # count all edges eligible for US market analysis
    total = len(eligible)
    # calculate proportion of collabs between artists with the same US availability status
    # formula: (US_US + nonUS_nonUS) / total eligible edges
    observed_same = (counts['US_US'] + counts['nonUS_nonUS']) / total if total else float("nan")

    summary = {
        "total_artists": G.number_of_nodes(),                                               # total number of artists
        "artists_with_market_info": len(known),                                             # artists with non-empty market sets
        "artists_available_in_US": sum(us_available.values()),                              # artists whose available markets include the US
        "artists_not_available_in_US": len(known) - sum(us_available.values()),             # artists with known market data that does not include the US
        "total_collaboration_edges": G.number_of_edges(),                                   # total number of edges in the graph
        "eligible_edges": total,                                                            # collabs between artists with known market info
        **counts,                                                                           # counts of collabs in each US availability category
        "same_US_status_rate": observed_same,                                               # observed proportion of matching status edges
    }
    # save to CSV
    pd.DataFrame([summary]).to_csv(output / "us_market_homophily.csv", index=False)

    # empty list to store community level homophily results
    records = []
    # iterate through communities from each partitioning algorithm
    for (algorithm, community_id), data in assignments.groupby(["algorithm", "community_id"]):
        # set of community members, retaining only artists IDs present in the graph
        members = set(data.artist_id) & set(G)
        # collab edges where both artists belong to the current community and both artists have known market info
        scoped = [(u, v) for u, v in G.subgraph(members).edges() if u in known and v in known]
        # counts the 3 US availability combinations within the community
        group_counts = mixing_counts(scoped, us_available)
        # count the number of eligible internal edges
        group_total = len(scoped)
        records.append({
            'algorithm': algorithm,                                                                                                         # partitioning algorithm used to detect the community
            'community_id': community_id,                                                                                                   # community ID
            'community_size': len(members),                                                                                                 # number of artists belonging to the community
            'artists_with_market_data': sum(n in known for n in members),                                                                   # number of artists in the community with known market info
            'artists_available_in_US': sum(us_available.get(n, False) for n in members),                                                    # number of artists with market data with US availability
            'artists_not_available_in_US': sum(n in known and not us_available[n] for n in members),                                        # number of artists with market data but without US availability
            'eligible_internal_edges': group_total,                                                                                         # number of eligible collabs within the community
            **group_counts,                                                                                                                 # counts of collabs in each US availability category within the community
            'same_US_status_rate': ((group_counts['US_US'] + group_counts['nonUS_nonUS']) / group_total if group_total else float('nan')),  # proportion of eligible internal edges connecting artists with the same US availability status
        })
    # save to CSV
    pd.DataFrame(records).to_csv(output / 'us_community_homophily.csv', index=False)
    print('US market homophily:', summary)

    
def main():
    parser = argparse.ArgumentParser()
    # path to artist CSV
    parser.add_argument("--artists", type=Path, default=ARTIST_CSV)
    # path to collaboration CSV
    parser.add_argument("--collaborations", type=Path, default=COLLAB_CSV)
    # path to community assignments CSV
    parser.add_argument("--assignments", type=Path, default=PARTITION_RESULTS)
    # path to output directory
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # load data
    artists = pd.read_csv(args.artists, dtype={"artist_id": str})
    collaborations = pd.read_csv(args.collaborations, dtype={"artist_1_id": str, "artist_2_id": str})
    assignments = pd.read_csv(args.assignments, dtype={"artist_id": str, "algorithm": str})
    markets = {r.artist_id: parse_markets(r.available_markets) for r in artists.itertuples(index=False)}
    genres = {r.artist_id: parse_genre(r.main_genre) for r in artists.itertuples(index=False)}

    # create graph from artists and collaborations
    G = create_graph(artists, collaborations)
    # get all edges
    edges = list(G.edges)
    # calculate total number of edges, number of market eligible edges and mean Jaccard similarity for the whole network
    overall = stats_for_edges(edges, markets)
    # count the number of artists in the network who have non-empty market sets
    overall['market_labelled_artists'] = sum(bool(markets[n]) for n in G)
    # save to CSV
    pd.DataFrame([overall]).to_csv(args.output / 'market_homophily.csv', index=False)

    # calculate genre homophily statistics for the entire network
    overall_genre = genre_stats_for_edges(edges, genres)
    # count the number of artists in the network who have known genres (not "unknown")
    overall_genre["genre_labelled_artists"] = sum(genres.get(n, "unknown") != "unknown" for n in G)
    # set the total edge count
    overall_genre["edges_total"] = len(edges)
    # save to CSV
    pd.DataFrame([overall_genre]).to_csv(args.output / "genre_homophily.csv", index=False)

    # empty list to store homophily stats for each community
    rows = []
    # iterate through all communities detected by each partitioning algorithm
    for (algorithm, community_id), data in assignments.groupby(["algorithm", "community_id"]):
        # get members from current community
        members = set(data.artist_id)
        # filter edges to include only those where both artists belong to the current community
        internal_edges = [(u, v) for u, v in G.subgraph(members).edges]
        # calculate market and genre homophily statistics for the current community
        market_result = stats_for_edges(internal_edges, markets)
        genre_result = genre_stats_for_edges(internal_edges, genres)
        rows.append({
            "algorithm": algorithm,                                                                 # partitioning algorithm used to detect the community
            "community_id": community_id,                                                           # community ID
            "size": len(members),                                                                   # number of artists in the community
            "market_labelled_artists": sum(bool(markets[n]) for n in members),                      # number of artists in the community with non-empty market sets
            "genre_labelled_artists": sum(genres.get(n, "unknown") != "unknown" for n in members),  # number of artists in the community with known genres
            **market_result,                                                                        # market homophily statistics for the community
            **genre_result,                                                                         # genre homophily statistics for the community
        })
    # save to CSV
    pd.DataFrame(rows).to_csv(args.output / 'community_homophily.csv', index=False)
    # calculate US market homophily for the entire network and for each community
    us_market_homophily(G, markets, assignments, args.output)
    print(f"results saved to {args.output}")


if __name__ == "__main__":
    main()        