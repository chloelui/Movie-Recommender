import csv
import numpy as np
from embeddings import cosine_similarity
from db import create_user, username_exists, get_user_id, log_recommendation, get_seen_movie_ids, get_disliked_movie_ids, record_feedback

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
def hybrid_score(target, movie, target_index, candidate_index, embeddings):
    sim = cosine_similarity(embeddings[target_index], embeddings[candidate_index])
    genre_score = genre_similarity(target, movie)
    return round(sim * 10 + genre_score * 0.5, 2)


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



# Load data and overview embeddings
movies = load_movies()
embeddings = np.load("data/movie_embeddings.npy")

# Access/create user profile
choice = input("New user or returning? (new/returning): ").strip().lower()

if choice == "new":
    username = input("Choose a username: ").strip()
    user_id = create_user(username)
    while user_id is None:
        username = input("That username is taken. Choose another: ").strip()
        user_id = create_user(username)
else:
    username = input("Enter your username: ").strip()
    user_id = get_user_id(username)
    while user_id is None:
        username = input("No account found with that name. Try again: ").strip()
        user_id = get_user_id(username)
seen_ids = get_seen_movie_ids(user_id)
disliked_ids = get_disliked_movie_ids(user_id)

title = input("Enter a movie name: ")
matches = [movie for movie in movies if movie["title"].lower() == title.lower()]

if not matches:
    print("Movie not found in dataset.")
    exit()

target_index = None
for i, movie in enumerate(movies):
    if movie["title"].lower() == title.lower():
        target_index = i                                                # Store target index for embedding lookup
        target = movie
        break
if target_index is None:
    print("Movie not found in dataset.")
    exit()

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
    (i, movie) for i, movie in enumerate(movies)
    if movie["id"] != target["id"] and movie["id"] not in seen_ids and movie["id"] not in disliked_ids
    and passes_filters(movie, include_genres, exclude_genres, include_actors, exclude_actors, min_year, max_year, min_rating)
]

# Provide recommendations from filtered candidates
recommendations = []
for i, movie in candidates:
    score = hybrid_score(target, movie, target_index, i, embeddings)
    recommendations.append((score, movie))

recommendations.sort(key=lambda x:x[0], reverse=True)                   # Sort recs by descending genre similarity

print(f"\nBecause you liked {target['title']}, we thought you might like:\n")
for score, movie in recommendations[:5]:
    log_recommendation(user_id, target, movie, score)                   # Log recommended movies
    print(f"{movie['title']} (genre similarity: {score})")

# Optionally collect user feedback
feedback_title = input("\nWant to rate one of these? Enter its title, or press Enter to skip: ").strip()
if feedback_title:
    rating_input = input("Rating (0-10): ").strip()
    liked_input = input("Did you like it? (y/n): ").strip().lower()
    match = next((m for _, m in recommendations[:5] if m["title"].lower() == feedback_title.lower()), None)
    if match:
        record_feedback(user_id, match["id"], match["title"], 
                        rating=float(rating_input) if rating_input else None, 
                        liked=(liked_input == "y") if liked_input else None)