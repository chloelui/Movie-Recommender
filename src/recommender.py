import csv

# Load movie dataset
def load_movies():
    with open("data/movies.csv", "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# Get number of shared genres b/w two movies
def genre_similarity(movie_a, movie_b):            
    genres_a = set(movie_a["genres"].split("|"))
    genres_b = set(movie_b["genres"].split("|"))

    return len(genres_a & genres_b) if genres_a and genres_b else 0


# Cast similarity
def cast_similarity(movie_a, movie_b):
    cast_a = set(movie_a["cast"].split("|"))
    cast_b = set(movie_b["cast"].split("|"))

    return len(cast_a & cast_b) if cast_a and cast_b else 0


# Add rating bonus to score
def recommendation_score(target, movie):
    genre_score = genre_similarity(target, movie)
    cast_score = cast_similarity(target, movie)
    rating = float(movie["vote_average"])                           # Higher-rated movies get higher score

    return round(genre_score * 2 + rating + cast_score * 0.5, 1)


# Turn comma-separated user input into set of lowercase strings
def parse_list_input(prompt):
    raw = input(prompt).strip()
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",")}


# Apply filters to whole movie list
def passes_filters(movie, include_genres, exclude_genres, include_actors, exclude_actors, min_year, max_year, min_rating):
    genres = {g.lower() for g in movie["genres"].split("|")} if movie["genres"] else set()
    cast = {c.lower() for c in movie["cast"].split("|")} if movie["cast"] else set()

    if include_genres and not (genres & include_genres):
        return False
    if exclude_genres and (genres & exclude_genres):
        return False
    if include_actors and not (cast & include_actors):
        return False
    if exclude_actors and (cast & exclude_actors):
        return False

    if movie["release_date"]:
        year = int(movie["release_date"][:4])
        if min_year and year < min_year:
            return False
        if max_year and year > max_year:
            return False

    if min_rating and float(movie["vote_average"]) < min_rating:
        return False

    return True




movies = load_movies()

title = input("Enter a movie name: ")
matches = [movie for movie in movies if movie["title"].lower() == title.lower()]

if not matches:
    print("Movie not found in dataset.")
    exit()

target = matches[0]

# Ask for users filters
print("\n(Press Enter to skip any filter)")
include_genres = parse_list_input("Genres you want (comma-separated): ")
exclude_genres = parse_list_input("Genres you don't want: ")
include_actors = parse_list_input("Actors/actresses you want: ")
exclude_actors = parse_list_input("Actors/actresses you don't want: ")

min_year_input = input("Earliest release year: ").strip()
max_year_input = input("Latest releast year: ").strip()
min_rating_input = input("Minimum rating (0-10): ").strip()

min_year = int(min_year_input) if min_year_input else None
max_year = int(max_year_input) if max_year_input else None
min_rating = float(min_rating_input) if min_rating_input else None

# Get all movies that match filters
candidates = [
    movie for movie in movies
    if movie["id"] != target["id"]
    and passes_filters(movie, include_genres, exclude_genres, include_actors, exclude_actors, min_year, max_year, min_rating)
]

# Provide recommendations from filtered candidates
recommendations = []
for movie in candidates:
    score = recommendation_score(target, movie)
    recommendations.append((score, movie))

recommendations.sort(key=lambda x:x[0], reverse=True)                   # Sort recs by descending genre similarity

print(f"\nBecause you liked {target['title']}, we thought you might like:\n")
for score, movie in recommendations[:5]:
    print(f"{movie['title']} (genre similarity: {score})")