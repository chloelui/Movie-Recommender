import csv
import numpy as np
import math
from embeddings import cosine_similarity
from db import create_user, username_exists, get_user_id, log_recommendation, get_watched_movie_ids, get_disliked_movie_ids, record_feedback
from llm_parser import parse_preferences

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


# With target movie 
def hybrid_score_with_anchor(target, movie, target_index, candidate_index, embeddings):
    sim = cosine_similarity(embeddings[target_index], embeddings[candidate_index])             # Semantic similarity
    genre_score = genre_similarity(target, movie)                                              # Genre similarity
    return round(sim * 10 + genre_score * 0.5, 2)


# No target movie
def hybrid_score_without_anchor(movie, include_genres):
    movie_genres = {g.lower() for g in movie["genres"].split("|")} if movie["genres"] else set()
    genre_match = len(movie_genres & include_genres) if include_genres else 0

    rating = float(movie["vote_average"]) if movie["vote_average"] else 0
    popularity = float(movie["popularity"]) if movie["popularity"] else 0

    return round(genre_match * 3 + rating * 0.5 + math.log1p(popularity) * 0.3, 2)


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
seen_ids = get_watched_movie_ids(user_id)
disliked_ids = get_disliked_movie_ids(user_id)


# Request user input
user_text = input("\nTell me what you're in the mood for: ")
parsed = parse_preferences(user_text)

include_genres = {g.lower() for g in parsed["include_genres"]}
exclude_genres = {g.lower() for g in parsed["exclude_genres"]}
include_actors = {g.lower() for g in parsed["include_actors"]}
exclude_actors = {g.lower() for g in parsed["exclude_actors"]}
min_year = parsed["min_year"]
max_year = parsed["max_year"]
min_rating = parsed["min_rating"]

target, target_index = None, None

title = parsed["target_movie"]                                          # If user mentioned specific movie
if title:
    import difflib
    matches = [(i, movie) for i, movie in enumerate(movies) if movie["title"].lower() == title.lower()]

    if not matches:
        all_titles = [movie["title"] for movie in movies]
        close = difflib.get_close_matches(title, all_titles, n=1, cutoff=0.6)
        if close:
            matches = [(i, movie) for i, movie in enumerate(movies) if movie["title"] == close[0]]

    if not matches:
        print(f"(Couldn't find '{title}' in the dataset — continuing without it.)")
    elif len(matches) == 1:
        target_index, target = matches[0]
    else:                                                               # Multiple movies with same name
        print("\nMultiple movies found with that title:")
        for idx, (i, movie) in enumerate(matches, start=1):
            year = movie["release_date"][:4] if movie["release_date"] else "N/A"
            print(f"{idx}. {movie['title']} ({year})")
        choice = input("Choose a number: ").strip()
        while not (choice.isdigit() and 1 <= int(choice) <= len(matches)):
            choice = input("Please enter a valid number: ").strip()
        target_index, target = matches[int(choice) - 1]


# Get all movies that match filters
candidates = [
    (i, movie) for i, movie in enumerate(movies)
    if (target is None or movie["id"] != target["id"])
    and movie["id"] not in seen_ids and movie["id"] not in disliked_ids
    and passes_filters(movie, include_genres, exclude_genres, include_actors, exclude_actors, min_year, max_year, min_rating)
]

# Provide recommendations from filtered candidates
recommendations = []
for i, movie in candidates:
    if target is not None:
        score = hybrid_score_with_anchor(target, movie, target_index, i, embeddings)
    else:
        score = hybrid_score_without_anchor(movie, include_genres)
    recommendations.append((score, movie))

recommendations.sort(key=lambda x: x[0], reverse=True)                   # Sort entire list of candidates by descending score


# Continue providing recommendations if user wants
offset = 0
shown_movies = []
while True:
    batch = recommendations[offset:offset + 5]                          # Go down list of recs

    if not batch:
        print("\nNo more recommendations available.")
        break

    if target is not None:
        print(f"\nBecause you liked {target['title']}, we thought you might like:\n")
    else:
        print("\nBased on what you're looking for, here are some picks:\n")

    for idx, (score, movie) in enumerate(batch, start=1):
        log_recommendation(user_id, target, movie, score)
        print(f"{idx}. {movie['title']} (score: {score})")

    shown_movies.extend(batch)
    offset += 5

    action = input("\nEnter a number for details, 'more' for new recommendations, or 'done' to finish: ").strip().lower()
    while action not in ("more", "done") and not (action.isdigit() and 1 <= int(action) <= len(batch)):
        action = input("Please enter a valid number, 'more', or 'done': ").strip().lower()

    while action.isdigit():
        selected = batch[int(action) - 1][1]
        year = selected["release_date"][:4] if selected["release_date"] else "N/A"
        print(f"\n{selected['title']} ({year}) — rating: {selected['vote_average']}")
        print(selected["overview"])
        action = input("\nEnter another number for details, 'more' for new recommendations, or 'done' to finish: ").strip().lower()
        while action not in ("more", "done") and not (action.isdigit() and 1 <= int(action) <= len(batch)):
            action = input("Please enter a valid number, 'more', or 'done': ").strip().lower()

    if action == "done":
        break
 

# Optionally collect user feedback
feedback_title = input("\nWant to rate one of these? Enter its title, or press Enter to skip: ").strip()
if feedback_title:
    rating_input = input("Rating (0-10): ").strip()
    liked_input = input("Did you like it? (y/n): ").strip().lower()
    match = next((m for _, m in shown_movies if m["title"].lower() == feedback_title.lower()), None)
    if match:
        record_feedback(user_id, match["id"], match["title"], 
                        watched=True,
                        rating=float(rating_input) if rating_input else None, 
                        liked=(liked_input == "y") if liked_input else None)
    else:
        print("That movie wasn't in the recommendations shown.")