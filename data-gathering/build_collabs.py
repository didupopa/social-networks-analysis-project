import csv
import json
from pathlib import Path
from itertools import combinations
from collections import defaultdict

from seed_artist_list import SEED_ARTISTS

# reuse the same project layout as data-gathering.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ARTIST_RELEASES_CACHE_DIR = CACHE_DIR / "artist-releases"
ALBUM_TRACKS_CACHE_DIR = CACHE_DIR / "album-tracks"
OUTPUT_CSV_COLLABS = DATA_DIR / "spotify_collaborations_2000_2022.csv"

START_YEAR = 2000
END_YEAR = 2022

# True  = only read release caches of the original seed artists
# False = read every cached artist, including the ones added by expand-artists.py
ONLY_SEED_ARTIST_RELEASES = True


# load JSON from disk (same helper as data-gathering.py)
def load_json(path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# extract the release year from a release date string
def get_release_year(release_date):
    if not release_date:
        return None

    try:
        return int(release_date[:4])
    except (ValueError, TypeError):
        return None


# artist_id -> artist_name, discovered from track credit data as we go
artist_names = {}

# unordered pair (artist_id_1, artist_id_2), sorted, -> number of shared songs
pair_counts = defaultdict(int)

# same pair -> set of every release year in which they have a shared song
pair_years = defaultdict(set)

# avoid double-counting a track if multiple seed artists reference the same album
processed_track_ids = set()

# every seed artist we already have a cached release list for
release_cache_files = sorted(ARTIST_RELEASES_CACHE_DIR.glob("*.json"))

if ONLY_SEED_ARTIST_RELEASES:
    seed_ids = set(SEED_ARTISTS.values())
    release_cache_files = [f for f in release_cache_files if f.stem in seed_ids]
print(f"Found {len(release_cache_files)} cached artist release files")

for index, release_cache_file in enumerate(release_cache_files, start=1):
    state = load_json(release_cache_file)

    if state is None or not state.get("completed"):
        print(f"Skipping {release_cache_file.name}: release list not fully downloaded")
        continue

    seed_artist_id = state.get("artist_id", release_cache_file.stem)
    releases = state.get("items", [])

    print(f"[{index}/{len(release_cache_files)}] {seed_artist_id}: {len(releases)} releases")

    for release in releases:
        release_id = release.get("id")

        if not release_id:
            continue

        release_year = get_release_year(release.get("release_date"))

        if release_year is None or release_year < START_YEAR or release_year > END_YEAR:
            continue

        tracks_cache_file = ALBUM_TRACKS_CACHE_DIR / f"{release_id}.json"
        tracks_state = load_json(tracks_cache_file)

        if tracks_state is None or not tracks_state.get("completed"):
            print(f"  Skipping release {release_id}: track list not cached/completed")
            continue

        for track in tracks_state.get("items", []):
            track_id = track.get("id")

            if not track_id or track_id in processed_track_ids:
                continue

            processed_track_ids.add(track_id)

            credited_artists = track.get("artists", [])

            # record artist names as we encounter them (no extra API calls needed)
            for artist in credited_artists:
                artist_id = artist.get("id")
                artist_name = artist.get("name")

                if artist_id and artist_id not in artist_names:
                    artist_names[artist_id] = artist_name

            # a track needs 2+ credited artists to represent a collaboration
            if len(credited_artists) < 2:
                continue

            credited_ids = sorted({
                artist.get("id")
                for artist in credited_artists
                if artist.get("id")
            })

            for artist_1_id, artist_2_id in combinations(credited_ids, 2):
                pair = (artist_1_id, artist_2_id)
                pair_counts[pair] += 1
                pair_years[pair].add(release_year)

print(f"Found {len(pair_counts)} unique collaborating pairs across {len(processed_track_ids)} unique tracks")

FIELDNAMES = [
    "artist_1_id",
    "artist_1_name",
    "artist_2_id",
    "artist_2_name",
    "shared_song_count",
    "first_collab_year",
    "last_collab_year",
    "collab_years",  # every year with a shared song, e.g. "2008;2015"
]

with OUTPUT_CSV_COLLABS.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
    writer.writeheader()

    # sorted by strongest collaborations first
    for (artist_1_id, artist_2_id), count in sorted(
        pair_counts.items(), key=lambda item: item[1], reverse=True
    ):
        years = sorted(pair_years[(artist_1_id, artist_2_id)])

        writer.writerow({
            "artist_1_id": artist_1_id,
            "artist_1_name": artist_names.get(artist_1_id, ""),
            "artist_2_id": artist_2_id,
            "artist_2_name": artist_names.get(artist_2_id, ""),
            "shared_song_count": count,
            "first_collab_year": years[0],
            "last_collab_year": years[-1],
            "collab_years": ";".join(str(y) for y in years),
        })

print()
print(f"CSV written to {OUTPUT_CSV_COLLABS.resolve()}")