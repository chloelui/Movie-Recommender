import csv
from tmdb_api import discover_movies

movies = []

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
        "overview"
    ])

    for movie in movies:
        writer.writerow([
            movie["id"],
            movie["title"],
            movie["release_date"],
            movie["vote_average"],
            movie["vote_count"],
            movie["popularity"],
            movie["overview"]
        ])


print(f"Saved {len(movies)} movies.")