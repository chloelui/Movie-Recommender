from tmdb_api import search_movies

# Hide API implementation
movies = search_movies("The Dark Knight")

for movie in movies:
    print(
        movie["id"],
        movie["title"],
        movie["release_date"]
    )