import csv
import math
import difflib
import numpy as np
from embeddings import cosine_similarity


def load_movies():
    with open("data/movies.csv", "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def load_embeddings():
    return np.load("data/movie_embeddings.npy")


def genre_similarity(movie_a, movie_b):
    genres_a = set(movie_a["genres"].split("|")) if movie_a["genres"] else set()
    genres_b = set(movie_b["genres"].split("|")) if movie_b["genres"] else set()
    return len(genres_a & genres_b) if genres_a and genres_b else 0


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


def hybrid_score_with_anchor(target, movie, target_index, candidate_index, embeddings):
    sim = cosine_similarity(embeddings[target_index], embeddings[candidate_index])
    genre_score = genre_similarity(target, movie)
    return round(sim * 10 + genre_score * 0.5, 2)


def hybrid_score_no_anchor(movie, include_genres):
    movie_genres = {g.lower() for g in movie["genres"].split("|")} if movie["genres"] else set()
    genre_match = len(movie_genres & include_genres) if include_genres else 0
    rating = float(movie["vote_average"]) if movie["vote_average"] else 0
    popularity = float(movie["popularity"]) if movie["popularity"] else 0
    return round(genre_match * 3 + rating * 0.5 + math.log1p(popularity) * 0.3, 2)


def find_movie_by_title(movies, title):
    """Exact match first, then fallback. Returns (index, movie) or (None, None)."""
    if not title:
        return None, None
    matches = [(i, m) for i, m in enumerate(movies) if m["title"].lower() == title.lower()]
    if matches:
        return matches[0]

    all_titles = [m["title"] for m in movies]
    close = difflib.get_close_matches(title, all_titles, n=1, cutoff=0.6)
    if close:
        matches = [(i, m) for i, m in enumerate(movies) if m["title"] == close[0]]
        if matches:
            return matches[0]
    return None, None


def generate_recommendations(movies, embeddings, target_index, target, filters, seen_ids, disliked_ids):
    include_genres = filters.get("include_genres", set())
    exclude_genres = filters.get("exclude_genres", set())
    include_actors = filters.get("include_actors", set())
    exclude_actors = filters.get("exclude_actors", set())
    min_year = filters.get("min_year")
    max_year = filters.get("max_year")
    min_rating = filters.get("min_rating")

    candidates = [
        (i, movie) for i, movie in enumerate(movies)
        if (target is None or movie["id"] != target["id"])
        and movie["id"] not in seen_ids and movie["id"] not in disliked_ids
        and passes_filters(movie, include_genres, exclude_genres, include_actors, exclude_actors, min_year, max_year, min_rating)
    ]

    recommendations = []
    for i, movie in candidates:
        if target is not None:
            score = hybrid_score_with_anchor(target, movie, target_index, i, embeddings)
        else:
            score = hybrid_score_no_anchor(movie, include_genres)
        recommendations.append((score, movie))

    recommendations.sort(key=lambda x: x[0], reverse=True)
    return recommendations