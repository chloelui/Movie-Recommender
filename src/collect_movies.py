import csv
import os
import time
from requests.exceptions import RequestException
from tmdb_api import discover_movies, get_genres, get_movie_credits

OUTPUT_PATH = "data/movies.csv"


def load_already_saved_ids(path):
    """If previous run already wrote some movies, read ids so don't refetch movies already there."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["id"] for row in reader}


def get_movie_credits_safe(movie_id, retries=3, backoff=5):
    """Wraps get_movie_credits with retries. Returns None (instead of an empty cast) if every attempt fails."""
    for attempt in range(1, retries + 1):
        try:
            return get_movie_credits(movie_id)
        except RequestException as e:
            print(f"  ! credits fetch failed for movie {movie_id} "
                  f"(attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(backoff * attempt)  
    print(f"  ! giving up on movie {movie_id} for now, will retry later")
    return None

movies = {}     # dict keyed by id to avoid duplicates

genres = get_genres()
genre_map = {genre["id"]: genre["name"] for genre in genres}

# Pull movies across several decades
date_ranges = [
    ("1874-01-01", "1899-12-31"),
    ("1900-01-01", "1929-12-31"),
    ("1930-01-01", "1949-12-31"),
    ("1950-01-01", "1969-12-31"),
    ("1970-01-01", "1979-12-31"),
    ("1980-01-01", "1999-12-31"),
    ("2000-01-01", "2009-12-31"),
    ("2010-01-01", "2019-12-31"),
    ("2020-01-01", "2026-12-31"),
]

MAX_PAGES_PER_BUCKET = 20  

for genre_id, genre_name in genre_map.items():
    for start, end in date_ranges:
        filters = {
            "with_genres": genre_id,
            "primary_release_date.gte": start,
            "primary_release_date.lte": end,
        }

        # First call to read total_pages/total_results for this genre + range
        first_page_movies, total_pages, total_results = discover_movies(1, **filters)

        if total_results == 0:
            continue

        pages_to_fetch = min(total_pages, MAX_PAGES_PER_BUCKET)

        print(f"{genre_name} {start[:4]}-{end[:4]}: {total_results} movies found, "
              f"fetching {pages_to_fetch} page(s)...")

        for movie in first_page_movies:
            movies[movie["id"]] = movie

        # Remaining pages after pg 1
        for page in range(2, pages_to_fetch + 1):
            page_movies, _, _ = discover_movies(page, **filters)
            for movie in page_movies:
                movies[movie["id"]] = movie     # deduplicate by id

movies = list(movies.values())

# Don't refetch movies from previous runs
already_saved_ids = load_already_saved_ids(OUTPUT_PATH)
movies_to_fetch = [m for m in movies if str(m["id"]) not in already_saved_ids]

if already_saved_ids:
    print(f"Found {len(already_saved_ids)} movies already saved from a previous "
          f"run. Skipping those, fetching the remaining {len(movies_to_fetch)}.")

os.makedirs("data", exist_ok=True)
file_exists = os.path.exists(OUTPUT_PATH)

failed_movies = []  # movies whose credits we couldn't fetch this pass

with open(OUTPUT_PATH, "a" if file_exists else "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow([
            "id", "title", "release_date", "vote_average", "vote_count",
            "popularity", "cast", "genres", "overview"
        ])

    def write_movie_row(movie, movie_cast):
        movie_genres = [genre_map[gid] for gid in movie["genre_ids"] if gid in genre_map]
        top_five_actors = [actor["name"] for actor in movie_cast["cast"][:5]]
        writer.writerow([
            movie["id"], movie["title"], movie["release_date"],
            movie["vote_average"], movie["vote_count"], movie["popularity"],
            '|'.join(top_five_actors), '|'.join(movie_genres), movie["overview"]
        ])

    # First pass
    total = len(movies_to_fetch)
    for i, movie in enumerate(movies_to_fetch, start=1):
        movie_cast = get_movie_credits_safe(movie["id"])
        if movie_cast is None:
            failed_movies.append(movie)
        else:
            write_movie_row(movie, movie_cast)

        if i % 50 == 0 or i == total:
            print(f"  ...processed {i}/{total} movies ({len(failed_movies)} failed so far)")
            file.flush()

    # Retry pass for anything failed first time
    if failed_movies:
        print(f"\nRetrying {len(failed_movies)} movie(s) that failed credit fetches...")
        still_failed = []
        for movie in failed_movies:
            movie_cast = get_movie_credits_safe(movie["id"], retries=3, backoff=10)
            if movie_cast is None:
                still_failed.append(movie)
            else:
                write_movie_row(movie, movie_cast)
        failed_movies = still_failed
        file.flush()

# Log anything still couldn't be fetched for rerunning
if failed_movies:
    with open("data/failed_movies.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title"])
        for movie in failed_movies:
            writer.writerow([movie["id"], movie["title"]])
    print(f"\n{len(failed_movies)} movie(s) still failed after retries. "
          f"Their ids are saved in data/failed_movies.csv — "
          f"rerun this script later and it will pick up where it left off "
          f"(already-saved movies are skipped automatically).")

print(f"\nDone. {len(movies_to_fetch) - len(failed_movies)} new movies saved this run.")