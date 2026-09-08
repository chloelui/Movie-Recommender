import csv
import json
import os

LINKS_PATH = "data/movielens/links.csv"
MOVIES_PATH = "data/movies.csv"
OUTPUT_PATH = "data/movielens_to_tmdb.json"

def build_tmdb_to_movielens_map(links_path=LINKS_PATH):
    """Reads links.csv and returns {tmdb_id (str) -> movielens_id (str)} where MovieLens movie has tmdbId."""
    mapping = {}
    with open(links_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tmdb_id = row["tmdbId"].strip()
            movielens_id = row["movieId"].strip()
            if tmdb_id:
                mapping[tmdb_id] = movielens_id
    return mapping

def report_coverage(mapping, movies_path=MOVIES_PATH):
    """Record fraction of TMDB catalog that has collaborative-filtering data available."""
    # Skip if movie catalog hasn't been created
    if not os.path.exists(movies_path):
        print(f"({movies_path} not found, skipping coverage report)")
        return

    with open(movies_path, newline="", encoding="utf-8") as f:
        your_ids = [row["id"] for row in csv.DictReader(f)]

    matched = sum(1 for mid in your_ids if mid in mapping)
    total = len(your_ids)
    pct = (matched / total * 100) if total else 0
    print(f"Coverage: {matched}/{total} of your movies ({pct:.1f}%) have a MovieLens id.")


if __name__ == "__main__":
    tmdb_to_ml = build_tmdb_to_movielens_map()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # Cache dict to disk as JSON for future scripts to load
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tmdb_to_ml, f)

    print(f"Linked {len(tmdb_to_ml)} MovieLens movies to TMDB ids.")
    print(f"Saved mapping to {OUTPUT_PATH}.")
    report_coverage(tmdb_to_ml)