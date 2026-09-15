import spotipy
import os
import csv
import json
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException
from dotenv import load_dotenv
from pathlib import Path

# load environment variables from .env
load_dotenv()

# redirect_uri and open_browser are not necessarily required for our data
# but was used only for testing purposes, since the data gathering step
# does not require user authentication
oauth = SpotifyOAuth(client_id=os.getenv("SPOTIFY_CLIENT_ID"),
                                               client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
                                               redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
                                               open_browser=True)
sp = spotipy.Spotify(auth_manager=oauth,
                     requests_timeout=20,
                     retries=0, # don't let spotipy automatically sleep after many hours when Spotify returns a long Retry-After value
                     status_retries=0) 
# define time range for data gathering & analysis
START_YEAR = 2000
END_YEAR = 2022

# include albums, singles and releases on which the artist appears
INCLUDE_GROUPS = "album" 

# define output file path for gathered data
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV_ARTISTS = DATA_DIR / "spotify_artists_2000_2022.csv"
OUTPUT_CSV_COLLABS = DATA_DIR / "spotify_collaborations_2000_2022.csv"

# define cache folders
CACHE_DIR = DATA_DIR / "cache"
ARTIST_CACHE_DIR = CACHE_DIR / "artists"
ARTIST_RELEASES_CACHE_DIR = CACHE_DIR / "artist-releases"
ALBUM_TRACKS_CACHE_DIR = CACHE_DIR / "album-tracks"

# seed artist list used to gather data from the Spotify API
SEED_ARTISTS = {
    "5 Seconds of Summer": "5Rl15oVamLq7FbSb0NNBNy"
}

# used for testing purposes, can be commented out
# print("Open URL in browser for authentication: ")
# print(oauth.get_authorize_url())
# print()

# load JSON from disk
def load_json(path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# save JSON to disk
def save_json(path, data):
    temp_path = path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    temp_path.replace(path)


# retrieve artist metadata cache
def get_artist_info(artist_id):
    cache_file = (ARTIST_CACHE_DIR / f"{artist_id}.json")

    cached = load_json(cache_file)

    if cached is not None:
        print(f"{artist_id} metadata cache")
        return cached

    print(f"{artist_id} metadata API call")
    artist_info = sp.artist(artist_id)
    save_json(cache_file, artist_info)
    return artist_info


# helper function to extract the release year from a release date string
def get_release_year(release_date):
    if not release_date:
        return None

    try:
        return int(release_date[:4])
    except (ValueError, TypeError):
        return None


# get all albums, singles and releases on which a specific artist appears
def get_all_artist_releases(artist_id):
    # check cache first
    cache_file = (ARTIST_RELEASES_CACHE_DIR / f"{artist_id}.json")

    state = load_json(cache_file)

    # no previous cache exists
    if state is None:
        state = {
            "artist_id": artist_id,
            "config": {
                "include_groups": INCLUDE_GROUPS,
            },
            "next_offset": 0,
            "completed": False,
            "items": []
        }

    # make sure we are not reusing cache generated with different config
    expected_config = {
        "include_groups": INCLUDE_GROUPS,
    }

    if state.get("config") != expected_config:
        raise RuntimeError(f"Cached release config for {artist_id} does not match INCLUDE_GROUPS={INCLUDE_GROUPS}. Please delete the cache file {cache_file} before continuing.")

    # entire release list is downloaded
    if state["completed"]:
        print(f"Releases cache for {artist_id} ({len(state['items'])} items)")
        return state["items"]

    # download releases from Spotify API
    print(f"Releases resume at offset {state['next_offset']} for {artist_id}")

    while not state["completed"]:
        offset = state["next_offset"]

        print(f"Requesting release offset {offset} for {artist_id}")

        results = sp.artist_albums(artist_id,
                                include_groups=INCLUDE_GROUPS,
                                limit=10,
                                offset=offset)

        page_items = results.get("items", [])

        state["items"].extend(page_items)

        # determine where to continue next time
        if results.get("next"):
            state["next_offset"] = offset + results.get("limit", 10)
        else:
            state["completed"] = True

        # checkpoint after every page
        save_json(cache_file, state)

    # remove releases with duplicate Spotify IDs
    unique_releases = {}

    # limitation here: if an artists releases the same album but like this
    # Album 1, Album 1(Deluxe), Album 1(Remastered)
    # the code will interpret it as 3 different albums
    for release in state["items"]:
        release_id = release.get("id")

        if release_id:
            unique_releases[release_id] = release

    # save deduplicated releases to cache
    state["items"] = list(unique_releases.values())
    save_json(cache_file, state)

    print(f"Releases completedL for {artist_id} ({len(state['items'])} items)")

    return state["items"]


# get every track from an album
# results are cached since several seed artists may show up on the same release
def get_all_album_tracks(album_id):
    cache_file = (ALBUM_TRACKS_CACHE_DIR / f"{album_id}.json")

    state = load_json(cache_file)

    # no cache yet
    if state is None:
        state = {
            "album_id": album_id,
            "next_offset": 0,
            "completed": False,
            "items": []
        }

    # already completed download
    if state["completed"]:
        print(f"Album tracks cache for {album_id} ({len(state['items'])} items)")
        return state["items"]

    # resume download from last offset
    print(f"Album tracks resume at offset {state['next_offset']} for {album_id}")

    while not state["completed"]:
        offset = state["next_offset"]
        results = sp.album_tracks(album_id, limit=50, offset=offset)

        page_items = results.get("items", [])

        state["items"].extend(page_items)

        # determine where to continue next time
        if results.get("next"):
            state["next_offset"] = offset + results.get("limit", 50)
        else:
            state["completed"] = True

        # save after each successful page
        save_json(cache_file, state)

    return state["items"]


# return True if artist_id is one of the credited Spotify artists on the track
def artist_is_on_track(track, artist_id):
    return any(
        artist.get("id") == artist_id
        for artist in track.get("artists", [])
    )


# construct all row entry values for one artist
# COLUMN NAMES:
# artist_id - Spotify artist ID
# artist_name - canonical name returned by Spotify
# main_genre - main genre returned by Spotify (although this feature is deprecated, might not show up for all artists)
# first_release_year - earliest release from 2000-2022 on which the artist is credited
# last_release_year - latest release from 2000-2022 on which the artist is credited
# number_of_albums - unique albums released by the artist from 2000-2022
# number_of_songs - unique Spotify tracks in 2000-2022 on which the artist is credited
# collaboration_count - numer of unique other artists appearing on those tracks
def build_artist_row(artist_id):
    artist_info = get_artist_info(artist_id)

    # get artist name
    artist_name = artist_info.get("name")

    # get artist main genre if available
    genres = artist_info.get("genres") or []
    main_genre = genres[0] if genres else ""

    # get discography for an artist
    releases = get_all_artist_releases(artist_id)
    unique_track_ids = set()
    unique_album_ids = set()
    collab_ids = set()
    activity_years = []

    # process releases
    for index, release in enumerate(releases, start=1):
        release_id = release.get("id")

        if not release_id:
            continue

        release_year = get_release_year(release.get("release_date"))

        if release_year is None:
            continue

        # keep track of releases in defined time frame
        if release_year < START_YEAR or release_year > END_YEAR:
            continue

        print(f"Release {index}/{len(releases)}: {release.get('name')}")

        # retrieve all tracks from the release
        tracks = get_all_album_tracks(release_id)

        # only keep tracks on which specific artist is actually credited
        artist_tracks = [
            track
            for track in tracks
            if artist_is_on_track(track, artist_id)
        ]

        if not artist_tracks:
            continue

        # release contributes to artist's years of activity if at least one relevant track exists
        activity_years.append(release_year)

        # count albums
        album_primary_artists_ids = {
            a.get("id")
            for a in release.get("artists", [])
        }

        if release.get("album_type") == "album" and artist_id in album_primary_artists_ids:
            unique_album_ids.add(release_id)

        # count songs and collabs
        for track in artist_tracks:
            track_id = track.get("id")

            if track_id:
                unique_track_ids.add(track_id)

            # every other credited artist is a collaborator
            for collaborator in track.get("artists", []):
                collaborator_id = collaborator.get("id")

                if collaborator_id and collaborator_id != artist_id:
                    collab_ids.add(collaborator_id)

    if activity_years:
        first_release_year = min(activity_years)
        last_release_year = max(activity_years)
    else:
        first_release_year = ""
        last_release_year = ""

    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "main_genre": main_genre,
        "first_release_year": first_release_year,
        "last_release_year": last_release_year,
        "number_of_albums": len(unique_album_ids),
        "number_of_songs": len(unique_track_ids),
        "collaboration_count": len(collab_ids)
        }

# CSV column names
FIELDNAMES_ARTISTS = [
    "artist_id",
    "artist_name",
    "main_genre",
    "first_release_year",
    "last_release_year",
    "number_of_albums",
    "number_of_songs",
    "collaboration_count"
]

# if CSV already exists, read already processed artists
PROCESSED_ARTIST_IDS = set()

if OUTPUT_CSV_ARTISTS.exists():
    with OUTPUT_CSV_ARTISTS.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            PROCESSED_ARTIST_IDS.add(row["artist_id"])

# append rows one artist at a time
# to make up for Spotify returning an error
file_exists = OUTPUT_CSV_ARTISTS.exists()

with OUTPUT_CSV_ARTISTS.open(
    "a",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(file, fieldnames=FIELDNAMES_ARTISTS)

    if not file_exists:
        writer.writeheader()

    total = len(SEED_ARTISTS)

    for index, (seed_name, artist_id) in enumerate(SEED_ARTISTS.items(), start=1):
        if artist_id in PROCESSED_ARTIST_IDS:
            print(f"[{index}/{total}] Skip {seed_name} which is already processed.")
            continue

        print(f"[{index}/{total}] Processing {seed_name}...")

        try:
            row = build_artist_row(artist_id)
            writer.writerow(row)
            # prevent Python from buffering the output and finished rowis written to disk immediately
            file.flush()

            print(f"{row['artist_name']}: {row['number_of_albums']} albums, {row['number_of_songs']} songs, {row['collaboration_count']} collaborations")
        except SpotifyException as error:
            print(f"Spotify error for {seed_name}: {error}")
            if error.http_status == 429:
                print("Rate limit exceeded.")
                print("Reason: ", error.reason)
                retry_after = None
                if error.headers:
                    retry_after = error.headers.get("Retry-After")
                if retry_after:
                    print(f"Retry-After: {retry_after} seconds")
                print("Stop here and run script again after waiting for the Retry-After duration.")
                break
            # other Spotify API error
            else:
                print(f"Spotify error for {seed_name}: {error}")
        except Exception as error:
            print(f"Unexpected error for {seed_name}: {error}")

print()
print(f"CSV written to {OUTPUT_CSV_ARTISTS.resolve()}")
