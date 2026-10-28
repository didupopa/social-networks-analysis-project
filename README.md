# Social Network Analysis project - Analysis and Visualization of Network Data for Spotify Artist Collaborations

This repository contains the code and data used to construct and analyze a Spotify artist collaboration network that our team worked on for the Social Network Analysis course (University of Groningen, Block 1A, 2026). Artists are represented as nodes and collaborations on shared songs are represented as connections between artists, with all data being in the 2000-2022 time range. The project collects artist and collaboration information using the Spotify Web API, expands the initial dataset through collaborating artists and applies network analysis techniques to examine the structure and characteristics of the resulting network.

## Repository structure

The repository is structured as follows:

```text
    social-networks-analysis-project
            |________________ data-analysis
            |                   |____ centrality_metrics.py
            |                   |____ chaperone_effect.py
            |                   |____ communities.py
            |                   |____ homophily.py
            |                   |____ k_components.py
            |                   |____ longitudinal_analysis.py
            |                   |____ network_analysis.py
            |                   |____ pagerank_analysis.py
            |                   |____ partitioning.py
            |                   |____ visualize_results.py
            |________________ data-gathering
            |                   |____ add_artist_album_titles.py
            |                   |____ add_artist_markets.py
            |                   |____ add_collab_albums_titles.py
            |                   |____ build_collabs.py
            |                   |____ data-gathering.py
            |                   |____ expand-artists.py
            |                   |____ seed_artist_list.py
            |                   |____ spotify_common.py
            |________________ data
            |                   |____ cache
            |                   |       |____ album-tracks
            |                   |       |____ artist-releases
            |                   |       |____ artists
            |                   |____ spotify_artists_2000_2022.csv
            |                   |____ spotify_collaborations_2000_2022.csv
            |________________ results
            |________________ .env
            |________________ .gitignore
            |________________ .venv
            |________________ README.md
            |________________ requirements.txt
```

The `data/` directory contains the data used in our analysis. The `cache/` directory stores information retrieved during data gathering, helping the scripts reuse previously collected API responses where supported by the implementation.

The `data-analysis/` directory provides the scripts conducting the analysis on the constructed network, while the `data-gathering/` one documents the scripts utilized in order to gather the data. Finally, the `results/` directory contains visualizations of our team's findings.

## Setting up the project

### 1. Create a Spotify App

The scripts inside the `data-gathering/` directory use the **Spotify Web API**. In order to obtain API credentials:

1. Visit the [Spotify developer portal](https://developer.spotify.com/dashboard/) and log in with your Spotify account.

2. Click the `Create an App` button. Enter any name and an optional description. Accept terms of service and click `Create`.

3. In the new App `Overview` screen, click on `Edit Settings` and scroll down to `Redirect URIs`. Add `http://127.0.0.1:1235/callback` (or any port number you would like). Hit the `Save` button and return to the App `Overview` screen.

4. Underneath the app name and description on the lefthand side, you will see a `Show Client Secret` link. Click on it and save both the **Client Secret** and the **Client ID** (check next step to where to save these values).

### 2. Configure environment variables

In the root of the repository, create a `.env` file and add the following:

```dotenv
SPOTIFY_CLIENT_ID=YOUR_CLIENT_ID
SPOTIFY_CLIENT_SECRET=YOUR_CLIENT_SECRET
SPOTIFY_REDIRECT_URI=YOUR_REDIRECT_URI
```

Replace the placeholders with your own credentials.

### 3. Create and activate a Python virtual environment

Run all commands in root of repository!

1. Create new Python virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment on macOS or Linux (first command) or on Windows Powershell (second command):

```bash
source .venv/bin/activate
```

```powershell
.venv\Scripts\Activate.ps1
```

3. To kill the virtual environment, run:
```bash
deactivate
```

### 4. Install required Python packages

After the virtual environment is running, install the required packages with the following command:

```bash
pip install -r requirements.txt
```
## Data

The project works with Spotify artist information and recorded collaborations between artists.

- **Artist data:** an artist level CSV dataset. Each row corresponds to an artist and includes an artist identifier, artist name, and collected summary attributes such as the number of albums, songs, and collaborations
- **Collaboration data:** a CSV dataset describing relationships between two artists. The collaboration records include both artists' identifiers and names and a `shared_song_count` indicating the number of songs they share
- **Cached API data:** reusable artist, release, and album track information kept under `data/cache/` during collection

For network analysis, **artists are nodes** and **collaborations are edges**. The `shared_song_count` can also be used as an edge weight when a particular analysis uses weighted relationships. The existence of a collaboration does not by itself imply a directed relationship.

The data reflects the artists and releases reached by the collection process and the availability of information through the Spotify API. It should not be interpreted as a complete dataset of all Spotify artists or collaborations.

## Data gathering

The collection workflow is implemented in `data-gathering/`. In the following sections, we document each script in terms of usability and running instructions.

### 1. Collect initial artist information

**Script:** `data-gathering/data-gathering.py`

The first step collects Spotify information for the initial set of artists, starting from the sample list of artists in `data-gathering/seed_artist_list.py`. This establishes the artist dataset used for subsequent collaboration discovery. The collection process uses the configured Spotify credentials and the project's cache directories.

```bash
cd data-gathering
python data-gathering.py
```

### 2. Construct the collaboration dataset

**Script:** `data-gathering/build_collabs.py`

This script processes collected artist and song information to identify songs involving multiple artists and construct collaboration records. These records provide the connections used to build the artist network, including counts of shared songs.

```bash
python build_collabs.py
```

### 3. Expand the artist dataset

**Script:** `data-gathering/expand-artists.py`

This script reads the collaboration CSV and identifies artists appearing in collaborations who have not yet been added to the artists CSV. It:

- Loads already processed artist IDs to avoid duplicate entries
- Checks that a collaboration meets the `MIN_SHARED_SONGS` threshold (set to `1` in the supplied version)
- Collects eligible new artist IDs and names
- Uses `build_artist_row()` from `spotify_common.py` to retrieve each new artist's information
- Appends the new artist records to the artists CSV, flushing after each successful write
- Reports Spotify API errors and stops when HTTP `429` indicates a rate limit, so the script can be run again later

```bash
python expand-artists.py
```

This step expands the *artist dataset* using already discovered collaborations; it does not itself generate additional collaboration edges. To discover connections involving newly added artists, the collaboration building stage must be run again if the collection workflow supports that iteration.

### 4. Add additional artist information

**Scripts:** `data-gathering/add_artist_album_titles.py`, `data-gathering/add_artist_markets.py`, `data-gathering/add_collab_albums_titles.py`

These scripts add additional information to the data CSVs from the cached data from the `Spotify Web API`: album titles for each individual artist, markets where their music is available and market count for each artist and titles of the albums on which each pair of artist has collaborated on.

```bash
python add_artist_album_titles.py
python add_artist_markets.py
python add_collab_albums_titles.py
```

### 5. Additional scripts

Script `data-gathering/spotify_common.py` contains helper methods used throughout the data gathering process.

## Network analysis

After gathering the data, the analysis scripts examine different structural properties of the artist collaboration network. The scripts below are described by their main analytical focus.

### Centrality — `centrality_metrics.py`

Identifies and compares the most important/influential artists in the collaboration network by examining different aspects of their position in the network, such as their connectivity, ability to connect different groups and connections to other influential artists.

```bash
python centrality_metrics.py
```

### Chaperone Effect — `chaperone_effect.py`

Investigates whether collaborating with highly connected artists helps other artists expand their collaboration networks. It examines whether artists gain more new collaborators after working with an influential, highly connected artists, referred to as a **chaperone**.

```bash
python chaperone_effect.py
```

### Communities — `communities.py`

Identifies groups of artists that are more closely connected to one another than to the rest of the network. Community detection helps reveal clusters of collaborating artists and provides a view of the network's broader group structure.

```bash
python communities.py
```

### Homophily — `homophily.py`

Examines whether collaborations tend to occur between artists sharing particular attributes. This helps assess the extent to which similarity is associated with connections in the artist network.

```bash
python homophily.py
```

### k-Components — `k_components.py`

Analyzes structural cohesion by identifying sets of artists with a given level of vertex connectivity. Unlike ordinary connected components, k-components help describe how resilient a group remains when artists (nodes) are removed.

```bash
python k_components.py
```

### Longitudinal Analysis — `longitudinal_analysis.py`

Analyzes how the local collaboration networks of the most connected artists change over time using their clustering coefficients. It investigates whether highly connected artists tend to collaborate with groups of artists who also collaborate with each other and how this changes from year to year.

```bash
python longitudinal_analysis.py
```

### Network Analysis — `network_analysis.py`

Provides ordinary network statistics insights about the constructed artist collaboration network such as number of vertices, number of edges, node degree, etc. These provide an initial overview of the network.

```bash
python network_analysis.py
```

### PageRank Analysis — `pagerank_analysis.py`

Looks into relative influence, authority and **prestige** of individuals or entities within a network. It is used in order to investigate whether more important nodes are likely to receive more incoming links from other nodes.

```bash
python pagerank_analysis.py
```

### Partitioning — `partitioning.py`

Examines ways to divide the network into groups or subnetworks, helping investigate its connectivity and internal organization.

```bash
python partitioning.py
```

### Result visualization — `visualize_results.py`

Produces visual representations of the computed results to help interpret and communicate findings from the network analyses.

```bash
python visualize_results.py
```

## Results

The `results/` directory is intended to hold outputs generated by the data analysis and visualization scripts. Each subfolder contains the visualization of each of the network analysis steps mentioned above.

## Reproducing the workflow

1. Configure the Spotify application credentials in `.env`.
2. Create the virtual environment and install dependencies from `requirements.txt`.
3. Prepare `data/cache/` subdirectories.
4. Run `data-gathering.py` to collect the initial artist information.
5. Run `build_collabs.py` to construct artist collaboration records.
6. Run `expand-artists.py` to add new artists discovered through those collaborations.
7. Run `add_artist_album_titles.py`, `add_artist_markets.py`, `add_collab_albums_titles.py` to add more information about the artists and their collaborations.
8. Use the resulting artist and collaboration datasets as inputs to the network analysis scripts.
9. Generate figures and inspect the saved analysis outputs in `results/`.