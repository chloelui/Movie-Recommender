from tmdb_api import search_movies, get_movie

query = input("Enter a movie name: ")

# Search all related movies
results = search_movies(query)

if not results:
    print("Movie not found.")
    exit()

print("\nSearch results:")

# Return top 5 search results
for i, movie in enumerate(results[:5]):
    print(f"{i + 1}. {movie["title"]}")

# Print info about selected movie
choice = int(input("\nChoose a movie number: ")) - 1
selected_movie = results[choice]
movie = get_movie(selected_movie["id"])

print("\n-------------------------")
print(movie["title"])
print("-------------------------")

print(f"Release date: {movie["release_date"]}")
print(f"Rating: {movie["vote_average"]}")
print(f"Runtime: {movie["runtime"]} minutes")

print("\nGenres:")

for genre in movie["genres"]:
    print(f"- {genre["name"]}")

print("\nOverview:")
print(movie["overview"])