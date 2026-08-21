import csv
from tmdb_api import discover_movies, get_genres, get_movie_credits

movies = []

# Convert genre ID to genre name
genres = get_genres()
genre_map = {genre["id"]:genre["name"] for genre in genres}

# Get movies across diff pages, sorted by popularity
for page in range(1,6):
    print(f"Downloading page {page}...")
    page_movies = discover_movies(page)
    movies.extend(page_movies)              # Append to list of movies


# Create local movie database
with open("data/movies.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "id",
        "title",
        "release_date",
        "vote_average",
        "vote_count",
        "popularity",
        "cast",
        "genres",
        "overview"
    ])

    for movie in movies:
        movie_genres = [genre_map[genre_id] for genre_id in movie["genre_ids"] if genre_id in genre_map]
        movie_cast = get_movie_credits(movie["id"])
        top_five_actors = [actor["name"] for actor in movie_cast["cast"][:5]]

        writer.writerow([
            movie["id"],
            movie["title"],
            movie["release_date"],
            movie["vote_average"],
            movie["vote_count"],
            movie["popularity"],
            '|'.join(top_five_actors),
            '|'.join(movie_genres),
            movie["overview"]
        ])


print(f"Saved {len(movies)} movies.")