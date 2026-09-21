import os
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

oauth = SpotifyOAuth(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI"),
    open_browser=True,
)
sp = spotipy.Spotify(
    auth_manager=oauth,
    requests_timeout=20,
    retries=0,  # don't let spotipy auto-sleep on long Retry-After values
    status_retries=0,
)

# define time range for data gathering & analysis
START_YEAR = 2000
END_YEAR = 2022

# include albums, singles and releases on which the artist appears
INCLUDE_GROUPS = "album"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_CSV_ARTISTS = DATA_DIR / "spotify_artists_2000_2022.csv"
OUTPUT_CSV_COLLABS = DATA_DIR / "spotify_collaborations_2000_2022.csv"

CACHE_DIR = DATA_DIR / "cache"
ARTIST_CACHE_DIR = CACHE_DIR / "artists"
ARTIST_RELEASES_CACHE_DIR = CACHE_DIR / "artist-releases"
ALBUM_TRACKS_CACHE_DIR = CACHE_DIR / "album-tracks"

FIELDNAMES_ARTISTS = [
    "artist_id",
    "artist_name",
    "main_genre",
    "first_release_year",
    "last_release_year",
    "number_of_albums",
    "number_of_songs",
    "collaboration_count",
]


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


# retrieve artist metadata, using cache
def get_artist_info(artist_id):
    cache_file = ARTIST_CACHE_DIR / f"{artist_id}.json"

    cached = load_json(cache_file)

    if cached is not None:
        print(f"{artist_id} metadata cache")
        return cached

    print(f"{artist_id} metadata API call")
    artist_info = sp.artist(artist_id)
    save_json(cache_file, artist_info)
    return artist_info


# extract the release year from a release date string
def get_release_year(release_date):
    if not release_date:
        return None

    try:
        return int(release_date[:4])
    except (ValueError, TypeError):
        return None


# get all albums, singles and releases on which a specific artist appears
def get_all_artist_releases(artist_id):
    cache_file = ARTIST_RELEASES_CACHE_DIR / f"{artist_id}.json"

    state = load_json(cache_file)

    if state is None:
        state = {
            "artist_id": artist_id,
            "config": {"include_groups": INCLUDE_GROUPS},
            "next_offset": 0,
            "completed": False,
            "items": [],
        }

    expected_config = {"include_groups": INCLUDE_GROUPS}

    if state.get("config") != expected_config:
        raise RuntimeError(
            f"Cached release config for {artist_id} does not match "
            f"INCLUDE_GROUPS={INCLUDE_GROUPS}. Please delete the cache "
            f"file {cache_file} before continuing."
        )

    if state["completed"]:
        print(f"Releases cache for {artist_id} ({len(state['items'])} items)")
        return state["items"]

    print(f"Releases resume at offset {state['next_offset']} for {artist_id}")

    while not state["completed"]:
        offset = state["next_offset"]

        print(f"Requesting release offset {offset} for {artist_id}")

        results = sp.artist_albums(
            artist_id, include_groups=INCLUDE_GROUPS, limit=10, offset=offset
        )

        page_items = results.get("items", [])
        state["items"].extend(page_items)

        if results.get("next"):
            state["next_offset"] = offset + results.get("limit", 10)
        else:
            state["completed"] = True

        save_json(cache_file, state)

    # remove releases with duplicate Spotify IDs
    unique_releases = {}

    for release in state["items"]:
        release_id = release.get("id")

        if release_id:
            unique_releases[release_id] = release

    state["items"] = list(unique_releases.values())
    save_json(cache_file, state)

    print(f"Releases completed for {artist_id} ({len(state['items'])} items)")

    return state["items"]


# get every track from an album (cached; shared across artists on the same release)
def get_all_album_tracks(album_id):
    cache_file = ALBUM_TRACKS_CACHE_DIR / f"{album_id}.json"

    state = load_json(cache_file)

    if state is None:
        state = {
            "album_id": album_id,
            "next_offset": 0,
            "completed": False,
            "items": [],
        }

    if state["completed"]:
        print(f"Album tracks cache for {album_id} ({len(state['items'])} items)")
        return state["items"]

    print(f"Album tracks resume at offset {state['next_offset']} for {album_id}")

    while not state["completed"]:
        offset = state["next_offset"]
        results = sp.album_tracks(album_id, limit=50, offset=offset)

        page_items = results.get("items", [])
        state["items"].extend(page_items)

        if results.get("next"):
            state["next_offset"] = offset + results.get("limit", 50)
        else:
            state["completed"] = True

        save_json(cache_file, state)

    return state["items"]


# return True if artist_id is one of the credited Spotify artists on the track
def artist_is_on_track(track, artist_id):
    return any(
        artist.get("id") == artist_id for artist in track.get("artists", [])
    )


# construct all row entry values for one artist (see FIELDNAMES_ARTISTS for columns)
def build_artist_row(artist_id):
    artist_info = get_artist_info(artist_id)

    artist_name = artist_info.get("name")

    genres = artist_info.get("genres") or []
    main_genre = genres[0] if genres else ""

    releases = get_all_artist_releases(artist_id)
    unique_track_ids = set()
    unique_album_ids = set()
    collab_ids = set()
    activity_years = []

    for index, release in enumerate(releases, start=1):
        release_id = release.get("id")

        if not release_id:
            continue

        release_year = get_release_year(release.get("release_date"))

        if release_year is None:
            continue

        if release_year < START_YEAR or release_year > END_YEAR:
            continue

        print(f"Release {index}/{len(releases)}: {release.get('name')}")

        tracks = get_all_album_tracks(release_id)

        artist_tracks = [
            track for track in tracks if artist_is_on_track(track, artist_id)
        ]

        if not artist_tracks:
            continue

        activity_years.append(release_year)

        album_primary_artists_ids = {
            a.get("id") for a in release.get("artists", [])
        }

        if release.get("album_type") == "album" and artist_id in album_primary_artists_ids:
            unique_album_ids.add(release_id)

        for track in artist_tracks:
            track_id = track.get("id")

            if track_id:
                unique_track_ids.add(track_id)

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
        "collaboration_count": len(collab_ids),
    }
