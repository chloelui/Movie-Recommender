import csv
from tmdb_api import discover_movies, get_genres, get_movie_credits

movies = {}  # use dict keyed by id to avoid duplicates

genres = get_genres()
genre_map = {genre["id"]: genre["name"] for genre in genres}

# Pull movies for each genre across several decades
date_ranges = [
    ("1980-01-01", "1999-12-31"),
    ("2000-01-01", "2009-12-31"),
    ("2010-01-01", "2019-12-31"),
    ("2020-01-01", "2025-12-31"),
]

for genre_id, genre_name in genre_map.items():
    for start, end in date_ranges:
        print(f"Downloading {genre_name} movies from {start[:4]}-{end[:4]}...")
        for page in range(1, 4):  # 3 pages per genre/date range
            page_movies = discover_movies(
                page,
                with_genres=genre_id,
                **{
                    "primary_release_date.gte": start,
                    "primary_release_date.lte": end,
                }
            )
            for movie in page_movies:
                movies[movie["id"]] = movie  # deduplicate by id

movies = list(movies.values())

# Create local movie database
with open("data/movies.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "id", "title", "release_date", "vote_average", "vote_count",
        "popularity", "cast", "genres", "overview"
    ])

    for movie in movies:
        movie_genres = [genre_map[gid] for gid in movie["genre_ids"] if gid in genre_map]
        movie_cast = get_movie_credits(movie["id"])
        top_five_actors = [actor["name"] for actor in movie_cast["cast"][:5]]

        writer.writerow([
            movie["id"], movie["title"], movie["release_date"],
            movie["vote_average"], movie["vote_count"], movie["popularity"],
            '|'.join(top_five_actors), '|'.join(movie_genres), movie["overview"]
        ])

print(f"Saved {len(movies)} movies.")