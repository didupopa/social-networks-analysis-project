import csv
import json

from itertools import combinations
from collections import defaultdict
from seed_artist_list import SEED_ARTISTS
from spotify_common import OUTPUT_CSV_COLLABS, ARTIST_RELEASES_CACHE_DIR, ALBUM_TRACKS_CACHE_DIR, START_YEAR, END_YEAR, load_json, get_release_year

# True  = only read release caches of the original seed artists
# False = read every cached artist, including the ones added by expand-artists.py
ONLY_SEED_ARTIST_RELEASES = True

# pair -> release_id -> (year, album_title)
pair_albums = defaultdict(dict)

# find artist release cache files
release_cache_files = sorted(ARTIST_RELEASES_CACHE_DIR.glob("*.json"))

if ONLY_SEED_ARTIST_RELEASES:
    seed_ids = set(SEED_ARTISTS.values())
    release_cache_files = [file for file in release_cache_files if file.stem in seed_ids]

print(f"reading {len(release_cache_files)} release caches")

# scan releases
for index, release_cache_file in enumerate(release_cache_files, start=1):
    state = load_json(release_cache_file)

    if state is None or not state.get("completed"):
        continue

    releases = state.get("items", [])
    print(f"[{index}/{len(release_cache_files)}] {release_cache_file.stem}")

    for release in releases:
        release_id = release.get("id")
        if not release_id:
            continue

        release_year = get_release_year(release.get("release_date"))

        if release_year is None:
            continue

        if not START_YEAR <= release_year <= END_YEAR:
            continue

        # only add real spotify album releases (not singles, etc.)
        if release.get("album_type") != "album":
            continue

        album_title = (release.get("name") or "").strip()
        if not album_title:
            continue

        # load tracks from cache
        tracks_cache_file = ALBUM_TRACKS_CACHE_DIR / f"{release_id}.json"
        tracks_state = load_json(tracks_cache_file)

        if tracks_state is None or not tracks_state.get("completed"):
            continue

        for track in tracks_state.get("items", []):
            credited_ids = sorted({artist.get("id") for artist in track.get("artists", []) if artist.get("id")})
            if len(credited_ids) < 2:
                continue

            # generate every collab pair on this track
            for artist_1_id, artist_2_id in combinations(credited_ids, 2):
                pair = (artist_1_id, artist_2_id)

                # use release_id to prevent the same spotify album from being added many times
                pair_albums[pair][release_id] = (release_year, album_title)


# read collab csv
with OUTPUT_CSV_COLLABS.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    fieldnames = list(reader.fieldnames)

if "album_titles" not in fieldnames:
    fieldnames.append("album_titles")

# add album titles to each collab edge
for row in rows:
    artist_1_id = row.get("artist_1_id")
    artist_2_id = row.get("artist_2_id")
    # undirected edges
    pair = tuple(sorted([artist_1_id, artist_2_id]))
    album_records = list(pair_albums.get(pair, {}).values())

    # sort in chronological order by year
    album_records.sort(key=lambda x: (x[0], x[1].lower()))

    # remove dup;icate titles
    titles = []
    seen_titles = set()
    for year, title in album_records:
        if title in seen_titles:
            continue
        seen_titles.add(title)
        titles.append(title)

    row["album_titles"] = json.dumps(titles, ensure_ascii=False)

# rewrite collab csv
temp_file = OUTPUT_CSV_COLLABS.with_name(OUTPUT_CSV_COLLABS.stem + "_temp.csv")
with temp_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# replace original file with temp file
temp_file.replace(OUTPUT_CSV_COLLABS)
print(f"updated collaboration csv with album titles: {OUTPUT_CSV_COLLABS.resolve()}")