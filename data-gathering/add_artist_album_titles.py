import csv
import json

from spotify_common import OUTPUT_CSV_ARTISTS, ARTIST_RELEASES_CACHE_DIR, START_YEAR, END_YEAR, load_json, get_release_year

# return spotify album titles for a given artist
# release is between START_YEAR and END_YEAR
# spotify says album_type == "album"
# artist is one of the primary credited album artists
def get_album_titles(artist_id):
    cache_file = ARTIST_RELEASES_CACHE_DIR / f"{artist_id}.json"
    state = load_json(cache_file)

    if state is None or not state.get("completed"):
        print(f"skipping {artist_id}: incomplete/missing cache")
        return None

    albums = []
    # prevent identical titles appearing more than once
    seen_titles = set()

    for release in state.get("items", []):
        release_year = get_release_year(release.get("release_date"))

        if release_year is None:
            continue

        if not START_YEAR <= release_year <= END_YEAR:
            continue

        # we want to keep only albums
        if release.get("album_type") != "album":
            continue

        # only count this as the artist's album when the artist
        # is one of the primary credited album artists
        primary_artist_ids = {a.get("id") for a in release.get("artists", []) if a.get("id")}

        if artist_id not in primary_artist_ids:
            continue

        title = (release.get("name") or "").strip()
        if not title:
            continue

        if title in seen_titles:
            continue

        seen_titles.add(title)

        albums.append((release_year, title))

    # chronoligcal order might be more useful than alphabetical
    albums.sort(key=lambda x: (x[0], x[1].lower()))

    return [title for year, title in albums]

# read existing artist csv
with OUTPUT_CSV_ARTISTS.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    fieldnames = list(reader.fieldnames)

# add the new column
if "album_titles" not in fieldnames:
    fieldnames.append("album_titles")

# add rows
total_rows = len(rows)
for index, row in enumerate(rows, start=1):
    artist_id = row["artist_id"]
    artist_name = row["artist_name"]
    print(f"[{index}/{total_rows}] processing {artist_name}...")

    titles = get_album_titles(artist_id)
    if titles is None:
        row["album_titles"] = ""
    else:
        # store as JSON inside the CSV cell
        row["album_titles"] = json.dumps(titles, ensure_ascii=False)

# write to temp file and replace original
temp_file = OUTPUT_CSV_ARTISTS.with_name(OUTPUT_CSV_ARTISTS.stem + "_temp.csv")

with temp_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

temp_file.replace(OUTPUT_CSV_ARTISTS)
print(f"updated {OUTPUT_CSV_ARTISTS} with album titles for {total_rows} artists")