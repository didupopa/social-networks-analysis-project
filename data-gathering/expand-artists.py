import csv
from spotipy.exceptions import SpotifyException

from spotify_common import (
    OUTPUT_CSV_ARTISTS,
    OUTPUT_CSV_COLLABS,
    FIELDNAMES_ARTISTS,
    build_artist_row,
)

# only pull in a new artist if they share at least this many songs with
# someone already in the network. Without this guardrail, a single feature
# verse from a huge-catalog artist could trigger a massive, unbounded crawl.
MIN_SHARED_SONGS = 1

# artist ids already present in the artists CSV -- skip these
processed_artist_ids = set()

if OUTPUT_CSV_ARTISTS.exists():
    with OUTPUT_CSV_ARTISTS.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            processed_artist_ids.add(row["artist_id"])

# collect every artist from the collaborations CSV that isn't already a node
new_artists = {}  # artist_id -> artist_name

with OUTPUT_CSV_COLLABS.open("r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        try:
            shared_song_count = int(row["shared_song_count"])
        except (KeyError, ValueError):
            shared_song_count = 0

        if shared_song_count < MIN_SHARED_SONGS:
            continue

        for id_col, name_col in (
            ("artist_1_id", "artist_1_name"),
            ("artist_2_id", "artist_2_name"),
        ):
            artist_id = row[id_col]
            artist_name = row[name_col]

            if artist_id and artist_id not in processed_artist_ids:
                new_artists[artist_id] = artist_name

print(f"Found {len(new_artists)} new artists to add (min {MIN_SHARED_SONGS} shared songs)")

file_exists = OUTPUT_CSV_ARTISTS.exists()

with OUTPUT_CSV_ARTISTS.open("a", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=FIELDNAMES_ARTISTS)

    if not file_exists:
        writer.writeheader()

    total = len(new_artists)

    for index, (artist_id, artist_name) in enumerate(new_artists.items(), start=1):
        print(f"[{index}/{total}] Processing {artist_name}...")

        try:
            row = build_artist_row(artist_id)
            writer.writerow(row)
            file.flush()

            print(
                f"{row['artist_name']}: {row['number_of_albums']} albums, "
                f"{row['number_of_songs']} songs, {row['collaboration_count']} collaborations"
            )
        except SpotifyException as error:
            print(f"Spotify error for {artist_name}: {error}")
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
            else:
                print(f"Spotify error for {artist_name}: {error}")
        except Exception as error:
            print(f"Unexpected error for {artist_name}: {error}")

print()
print(f"CSV written to {OUTPUT_CSV_ARTISTS.resolve()}")