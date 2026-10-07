import csv
import json

from spotify_common import OUTPUT_CSV_ARTISTS, ARTIST_RELEASES_CACHE_DIR, START_YEAR, END_YEAR, load_json, get_release_year

def get_cached_markets(artist_id):
    cache_file = ARTIST_RELEASES_CACHE_DIR / f"{artist_id}.json"
    state = load_json(cache_file)

    if state is None or not state.get("completed"):
        return None

    markets = set()
    market_field_found = False

    for release in state.get("items", []):
        release_year = get_release_year(release.get("release_date"))
        if release_year is None:
            continue

        if not START_YEAR <= release_year <= END_YEAR:
            continue

        if "available_markets" in release:
            market_field_found = True
            markets.update(release["available_markets"] or [])

    # distinguish the following
    # None -> field unavailable in cached data
    # [] -> field existed but contained no markets
    if not market_field_found:
        return None

    return sorted(markets)


# read csv
with OUTPUT_CSV_ARTISTS.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    fieldnames = list(reader.fieldnames)

for column in ("available_markets", "market_count"):
    if column not in fieldnames:
        fieldnames.append(column)

# add cached market data
for row in rows:
    artist_id = row["artist_id"]
    markets = get_cached_markets(artist_id)
    if markets is None:
        # the field was not available in the cached data
        row["available_markets"] = ""
        row["market_count"] = ""
    else:
        row["available_markets"] = json.dumps(markets, ensure_ascii=False)
        row["market_count"] = len(markets)

# write updated csv
temp_file = OUTPUT_CSV_ARTISTS.with_name(OUTPUT_CSV_ARTISTS.stem + "_temp.csv")
with temp_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# replace original csv with updated csv
temp_file.replace(OUTPUT_CSV_ARTISTS)
print(f"updated {OUTPUT_CSV_ARTISTS.resolve()} with cached market data")