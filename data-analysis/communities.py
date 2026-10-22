import argparse
import pandas as pd
import networkx as nx

from pathlib import Path
from networkx.algorithms.community import k_clique_communities

# paths
ROOT_DIR = Path(__file__).parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIST_CSV = DATA_DIR / "spotify_artists_2000_2022.csv"
COLLAB_CSV = DATA_DIR / "spotify_collaborations_2000_2022.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "communities"


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


# helper function to record group common information
# used for cliques, k-cores, k-components and k-clique communities
def group_record(G, members, group_type, group_id, k=None):
    # sort artists alphabetically by name, then by id
    members = sorted(members, key=lambda n: (G.nodes[n].get('name', ''), n))
    # create subgraph containing only the artists in the group and the collaboration edges between them
    sub = G.subgraph(members)
    # calculate maximum possible number of edges among the group members in an undirected graph - n(n-1)/2
    possible = len(members) * (len(members) - 1) // 2
    
    return {
        "type": group_type,                                                         # group classification: clique, k-core, k-component, k-clique community, near clique
        "group_id": group_id,                                                       # identifier of the group
        "k": k,                                                                     # relevant k value if applicable
        "size": len(members),                                                       # number of artists in the group
        "edges": sub.number_of_edges(),                                             # number of collaboration edges among the group members
        "density": round(nx.density(sub), 5) if len(members) > 1 else 0,            # proportion of possible edges that actually exist
        "missing_edges": possible - sub.number_of_edges(),                          # number of missing edges needed to make the group a clique
        "total_shared_songs": sum(e["weight"] for *_, e in sub.edges(data=True)),   # sum of collaboration weights within the group
        "artist_ids": ";".join(members),                                            # semicolon separated list of artist IDs in the group
        "artist_names": "; ".join(G.nodes[n].get("name", n) for n in members),      # semicolon separated list of artist names in the group
    }


# helper function to find a missing edge near clique candidates from maximal cliques
def near_cliques_one_missing_edge(G, maximal_cliques, min_size=5):
    # empty set to hold near clique candidates
    candidates = set()
    # go through all maximal cliques in the graph
    for clique in maximal_cliques:
        # skip cliques that are too smal to form a near clique of desired size
        # even after adding one artist
        if len(clique) + 1 < min_size:
            continue
        # convert to set so operations are faster
        members = set(clique)
        # retrieve neighbors of each artist in the clique
        # combine all these neighbors into one set
        # remove artists who are already part of the clique
        possible_additions = set().union(*(set(G.neighbors(v)) for v in clique)) - members
        # check each potential additional artist
        for outsider in possible_additions:
            # count how many clique members collaborate with the outside artist
            links = sum(G.has_edge(outsider, v) for v in clique)
            # check whether outside artist is connected to all clique members except one
            if links == len(clique) - 1:
                # combine clique members with the outside artist into one group
                # and check if it satisfies the minimum size requirement
                group = frozenset(members | {outsider})
                if len(group) >= min_size:
                    candidates.add(group)

    # create list to store the final near clique candidates
    # after filtering out redundant groups
    maximal = []
    for candidate in sorted(candidates, key=lambda c: (-len(c), tuple(sorted(c)))):
        if not any(candidate < accepted for accepted in maximal):
            maximal.append(candidate)
    return maximal


# helper function to save group records to a CSV file
def save_records(records, output_path):
    # define CSV column names
    columns = ["type", "group_id", "k", "size", "edges", "density", "missing_edges", "total_shared_songs", "artist_ids", "artist_names"]
    df = pd.DataFrame(records, columns=columns)
    # sort data frame by number of artists then by total number of shared songs, both in descending order
    if not df.empty:
        df = df.sort_values(["size", "total_shared_songs"], ascending=[False, False])
    # save to CSV file
    df.to_csv(output_path, index=False)
    return df


# helper function to identify potential bridge artists using shortest paths and k-cliques communities
def find_bridge_artists(G, communities, k, betweenness, articulation_points):
    # create dictionary where each artist is associated with an initially empty set of community IDs
    memberships = {node: set() for node in G.nodes}
    # add community IDs to the membership set of each artist belonging to that community
    for community_id, group in enumerate(communities, 1):
        for node in group:
            memberships[node].add(community_id)

    # empty list to store bridge related metrics for each artist
    records = []
    # iterate through all artists in the graph
    for node in G.nodes:
        # get communities to which the current artist belongs to
        own = memberships[node]
        # store communities containing at least one direct neighbor of the current artist
        neighbor_communities = set()
        # store neighboring artists belonging to at least one community that the current artist does not belong to
        external_neighbors = set()
        # iterate through all direct neighbors of the current artist
        for neighbor in G.neighbors(node):
            # add collaborator's community memberships to the set of neighboring communities
            neighbor_communities.update(memberships[neighbor])
            # an external neighbor reaches at least one community not containing node
            if memberships[neighbor] - own:
                external_neighbors.add(neighbor)

        # combine artist's own communities with those reached through direct collaborators
        touched = own | neighbor_communities
        records.append({
            "artist_id": node,                                                      # artist identifier
            "artist_name": G.nodes[node].get("name", str(node)),                    # artist name
            "k": k,                                                                 # k-clique community level
            "degree": G.degree(node),                                               # number of direct collaborators
            "betweenness_centrality": round(betweenness[node], 6),                  # fraction of shortest paths passing through the artist
            "articulation_point": node in articulation_points,                      # whether removing the artist increases the number of connected components
            "member_communities": len(own),                                         # number of communities containing the artist
            "communities_touched": len(touched),                                    # number of communities the artist belongs to or directly reaches
            "external_communities_reached": len(touched - own),                     # number of communities reached that do not contain the artist
            "external_neighbors": len(external_neighbors),                          # number of direct collaborators belonging to at least one external community
            "community_ids_touched": ";".join(map(str, sorted(touched))),           # IDs of all touched communities
        })

    return records


def main():
    parser = argparse.ArgumentParser()
    # path to artist CSV
    parser.add_argument("--artists", default=ARTIST_CSV)
    # path to collaboration CSV
    parser.add_argument("--collaborations", default=COLLAB_CSV)
    # path to output directory for results
    parser.add_argument("--output", default=OUTPUT_DIR)
    # minimum size of maximal cliques to consider, default is 3
    parser.add_argument("--min-clique-size", type=int, default=3)
    # minimum size of near clique candidates, default is 5
    parser.add_argument("--min-near-clique-size", type=int, default=5)
    # computationally expensive full network k-components can be skipped with this flag
    parser.add_argument("--skip-k-components", action="store_true")
    args = parser.parse_args()

    # load data, create graph and output directory
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    artists = pd.read_csv(args.artists)
    collaborations = pd.read_csv(args.collaborations)
    G = create_graph(artists, collaborations)
    print(f"graph constructed: {G.number_of_nodes()} artists, {G.number_of_edges()} collaborations")
    # result: connected components: 25
    print(f"connected components: {nx.number_connected_components(G)}")

    # maximal cliques
    maximal_cliques = list(nx.find_cliques(G))
    # get cliques from graph that are at least the minimum size specified by the user
    cliques = [set(c) for c in maximal_cliques if len(c) >= args.min_clique_size]
    # save to CSV
    clique_df = save_records([group_record(G, c, "clique", i) for i, c in enumerate(cliques, 1)], output / "cliques.csv")
    # result: maximal cliques of size >= 3: 1320
    print(f"maximal cliques of size >= {args.min_clique_size}: {len(clique_df)}")

    # k-cores: write the coreness of each artist, plus separate connected groups
    # calculate core number for each artist (largerst k for which the artist belongs to a k-core)
    cores = nx.core_number(G)
    # save core numbers to CSV
    pd.DataFrame([{"artist_id": n, "artist_name": G.nodes[n]["name"], "core_number": k} for n, k in cores.items()]).sort_values("core_number", ascending=False).to_csv(output / "artist_core_numbers.csv", index=False)

    # empty list to store information about connected components of k-cores
    core_records = []
    # highest core number in the graph, default to 0 if no cores exist
    max_k = max(cores.values(), default=0)
    # iterate through k values starting at 2 and ending at the maximum core number
    for k in range(2, max_k + 1):
        # extract k-core of the graph
        sub = nx.k_core(G, k=k, core_number=cores)
        # k-core is not necessarily a connected component
        # this identifies the connected components within the k-core subgraph
        components = list(nx.connected_components(sub))
        # save statistics for each connected component of the current k-core
        for i, c in enumerate(components, 1):
            core_records.append(group_record(G, c, "k_core_component", i, k))
    core_df = save_records(core_records, output / "k_core_components.csv")
    # result: maximum core number: 9, k-core components saved: 27
    print(f"maximum core number: {max_k}, k-core components saved: {len(core_df)}")

    # k-components: maximal subgraphs with vertex connectivity at least k
    # this hasn't been run since it takes a lot of time for the entire network
    # additional k-component analysis on a subgraph can be found in k_components.py
    if not args.skip_k_components:
        print("computing full network k-components")
        # compute k-components at different levels of vertex connectivity
        k_components_by_level = nx.k_components(G)
        # create list for group statistics and a dictionary to track the highest k-component level of every artist
        k_component_records = []
        highest_k = {node: 0 for node in G.nodes}
        # iterate through k-component levels in ascending order
        for k, groups in sorted(k_components_by_level.items()):
            # record each k-component's
            for i, members in enumerate(groups, 1):
                k_component_records.append(group_record(G, members, "k_component", i, k))
                # update the highest k-component level for every group member
                for node in members:
                    highest_k[node] = max(highest_k[node], k)
            print(f"{k}-components: {len(groups)}")
        # save sttaistics for all identified k-components to CSV
        save_records(k_component_records, output / "k_components.csv")
        pd.DataFrame([
            {"artist_id": node, "artist_name": G.nodes[node]["name"], "highest_k_component": highest_k[node]} for node in G.nodes
        ]).sort_values("highest_k_component", ascending=False).to_csv(output / "artist_k_component_levels.csv", index=False)
        print(f"k-components saved: {len(k_component_records)}")

    # k-clique communities (overlapping communities are possible)
    
    # get size of the largest maximal clique
    max_clique_size = max(len(c) for c in maximal_cliques)
    # initialize lists for community statistics and bridge related metrics
    community_records = []
    bridge_records = []
    # calculate normalized betweenness centrality for each artist
    betweenness = nx.betweenness_centrality(G, normalized=True, weight=None)
    # identify all artists whose removal would increase the number of connected components
    articulation_points = set(nx.articulation_points(G))
    # iterate through k values from 3 to the largest clique size
    for k in range(3, max_clique_size + 1):
        # identify k-clique communities at the current k level
        communities = list(k_clique_communities(G, k, cliques=maximal_cliques))
        # save statistics for each identified community
        for i, c in enumerate(communities, 1):
            community_records.append(group_record(G, c, "k_clique_community", i, k))
        # calculate bridge related metrics for each artist at the current k level and adds them to the existing records
        bridge_records.extend(find_bridge_artists(G, communities, k, betweenness, articulation_points))
        # result: 
        # 3-clique communities: 227
        # 4-clique communities: 159
        # 5-clique communities: 76
        # 6-clique communities: 24
        # 7-clique communities: 12
        # 8-clique communities: 8
        # 9-clique communities: 3
        # 10-clique communities: 2
        print(f"{k}-clique communities: {len(communities)}")
    # save to CSV
    save_records(community_records, output / "k_clique_communities.csv")

    # potential bridge artists: shortest path betweenness and cross community links
    # sort by k level in ascending order, then by number of external communities reached in descending order and then by betweenness centrality in descending order
    bridges_df = pd.DataFrame(bridge_records).sort_values(["k", "external_communities_reached", "betweenness_centrality"], ascending=[True, False, False])
    # save to CSV
    bridges_df.to_csv(output / "bridge_artists_all_k.csv", index=False)
    print(f"bridge artists saved: {len(bridges_df)}")

    # approximate near cliques with one missing edge
    near = near_cliques_one_missing_edge(G, maximal_cliques, min_size=args.min_near_clique_size)
    # save to CSV
    near_df = save_records([group_record(G, c, "one_missing_edge_near_clique", i) for i, c in enumerate(near, 1)], output / "near_cliques.csv")
    # result: near clique candidates (>= 5 artists): 1167
    print(f"near clique candidates (>= {args.min_near_clique_size} artists): {len(near_df)}")

    # print the five largest groups identified by each method
    for label, frame in [("largest cliques", clique_df), ("largest k-cores", core_df), ("Largest k-cliques", pd.DataFrame(community_records)), ("largest near cliques", near_df)]:
        print(f"\n{label}:")
        for _, row in frame.sort_values(["size", "total_shared_songs"], ascending=[False, False]).head(5).iterrows():
            prefix = f"k={int(row['k'])}, " if pd.notna(row['k']) else ''
            print(f"{prefix}size={row['size']}, density={row['density']}, artists={str(row['artist_names'])[:180]}")
    # results:
    # largest cliques:
    # size=10, density=1.0, artists=Bas; Buddy; Dreamville; Guapdad 4000; JID; Jace; Mez; Reese LAFLARE; Ski Mask The Slump God; Smokepurpp
    # size=10, density=1.0, artists=Destra; Dovey Magnum; Lady Leshurr; Lisa Mercedez; London Hill; Nicki Minaj; Pamputtae; Patrice Roberts; Skeng; Spice
    # size=9, density=1.0, artists=Ace Hood; Ball Greezy; Billy Blue; Bred; Brisco; Flo Rida; Pitbull; Red Eyezz; Rick Ross
    # size=8, density=1.0, artists=Bas; Buddy; Cozz; Dreamville; EARTHGANG; J. Cole; JID; Smino
    # size=8, density=1.0, artists=Bas; Buddy; Cozz; Dreamville; EARTHGANG; Guapdad 4000; JID; Smino

    # largest k-cores:
    # k=2, size=1336, density=0.00455, artists=$tupid Young; 112; 2 Chainz; 21 Savage; 22Bullets; 2Baba; 2Pac; 3 Are Legend; 3OH!3; 3rd Generation (Bereal Family); 50 Cent; 6LACK; 6ix9ine; 7 Days Of Funk; 7th Heaven; ?uestlove;
    # k=3, size=803, density=0.00964, artists=$tupid Young; 112; 2 Chainz; 21 Savage; 22Bullets; 2Pac; 3OH!3; 50 Cent; 6LACK; 7 Days Of Funk; A Boogie Wit da Hoodie; A$AP Ferg; A$AP Rocky; AFROJACK; AJR; AKA; Aaron Shaw; Ab-So
    # k=4, size=495, density=0.01883, artists=2 Chainz; 21 Savage; 22Bullets; 50 Cent; 6LACK; 7 Days Of Funk; A$AP Ferg; A$AP Rocky; AFROJACK; Ab-Soul; Ace Hood; Action Bronson; Akbar V; Akon; Albert Harvey; Alberto Stylee; Al
    # k=5, size=312, density=0.0344, artists=2 Chainz; 50 Cent; 6LACK; A$AP Rocky; AFROJACK; Ab-Soul; Ace Hood; Action Bronson; Akbar V; Akon; Alberto Stylee; Alex Kyza; Alicia Keys; Anderson .Paak; Anitta; Anuel AA; Arcángel
    # k=6, size=229, density=0.05052, artists=2 Chainz; 50 Cent; 6LACK; A$AP Rocky; AFROJACK; Ab-Soul; Ace Hood; Action Bronson; Akon; Alex Kyza; Anderson .Paak; Anuel AA; Arcángel; Ari Lennox; Ariana Grande; Baby Rasta y Grin

    # Largest k-cliques:
    # k=3, size=747, density=0.01017, artists=112; 2 Chainz; 21 Savage; 2Pac; 3rd Generation (Bereal Family); 50 Cent; 6LACK; 7 Days Of Funk; 7th Heaven; A Boogie Wit da Hoodie; A$AP Ferg; A$AP Rocky; AFROJACK; AJR; Aaron Shaw
    # k=4, size=144, density=0.07071, artists=2 Chainz; 2Pac; 50 Cent; A$AP Rocky; AFROJACK; Ace Hood; Akon; Alicia Keys; Anderson .Paak; Ariana Grande; BJ The Chicago Kid; Ball Greezy; Bebe Rexha; Beyoncé; Billy Blue; Birdman
    # k=4, size=36, density=0.27778, artists=2 Chainz; 6LACK; Ari Lennox; Baby Rose; Baby Tate; Bas; Buddy; Childish Major; Cozz; DaBaby; DaVionne; Dreamville; Dreezy; EARTHGANG; G Perico; Girl Talk; Guapdad 4000; J. Cole; JI
    # k=4, size=32, density=0.22581, artists=Ab-Soul; Allen Stone; Anderson .Paak; Ben Bridwell; Buffalo Madonna; Carla Morrison; Chance the Rapper; DJ Premier; Ed Sheeran; Eighty4 Fly; Eric Nally; Evan Roman; Grandmaster Caz
    # k=5, size=30, density=0.35862, artists=2 Chainz; 6LACK; Ari Lennox; Baby Tate; Bas; Buddy; Childish Major; Cozz; DaVionne; Dreamville; Dreezy; EARTHGANG; Guapdad 4000; J. Cole; JID; Jace; Landstrip Chip; Lute; Mez; Omen

    # largest near cliques:
    # size=9, density=0.97222, artists=Bas; Buddy; Cozz; Dreamville; EARTHGANG; Guapdad 4000; J. Cole; JID; Smino
    # size=9, density=0.97222, artists=Jasper Dolphin; L-Boy; Lee Spielman; Left Brain; Lucas.; Na-Kel Smith; Taco; Taco; Tyler, The Creator
    # size=8, density=0.96429, artists=Bas; Buddy; Dreamville; EARTHGANG; Guapdad 4000; JID; Mez; Smino
    # size=8, density=0.96429, artists=Ari Lennox; Bas; Buddy; Dreamville; EARTHGANG; Guapdad 4000; JID; Smino
    # size=8, density=0.96429, artists=Bas; Buddy; Cozz; Dreamville; Guapdad 4000; JID; Mez; Smino
    
    # confirmation message that everything has been saved to CSV files in the output directory
    print(f'saved CSV results to: {output.resolve()}')


if __name__ == "__main__":
    main()