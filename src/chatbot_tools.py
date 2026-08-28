from db import log_recommendation, record_feedback
from recommender_engine import find_movie_by_title, generate_recommendations


def build_tools(session):
    """Returns the four callable actions, all closed over user's session state
    (movies, embeddings, user_id, what's been shown so far)."""

    def get_recommendations(target_movie="", include_genres=None, exclude_genres=None, include_actors=None,
                            exclude_actors=None, min_year=None, max_year=None, min_rating=None):
        movies = session["movies"]
        embeddings = session["embeddings"]

        match = find_movie_by_title(movies, target_movie) if target_movie else \
            {"status": "no_query", "index": None, "movie": None, "matched_title": None, "queried_title": None}

        target_index, target = match["index"], match["movie"]

        # Report fuzzy matches back and let recommendations continue on genre/filter alone this turn
        if match["status"] == "fuzzy":
            target_index, target = None, None

        filters = {
            "include_genres": {g.lower() for g in (include_genres or [])},
            "exclude_genres": {g.lower() for g in (exclude_genres or [])},
            "include_actors": {a.lower() for a in (include_actors or [])},
            "exclude_actors": {a.lower() for a in (exclude_actors or [])},
            "min_year": min_year,
            "max_year": max_year,
            "min_rating": min_rating,
        }

        scored = generate_recommendations(
            movies, embeddings, target_index, target, filters,
            session["seen_ids"], session["disliked_ids"]
        )

        session["last_scored"] = scored
        session["last_offset"] = 0
        session["last_target"] = target

        result = _take_next_batch(session)
        result["title_match"] = {
            "status": match["status"],
            "queried_title": match["queried_title"],
            "matched_title": match["matched_title"],
        }
        return result


    def more_recommendations():
        if "last_scored" not in session:
            return {"error": "No previous recommendations to continue from."}
        return _take_next_batch(session)


    def get_movie_details(title):
        movies = session["movies"]
        # Try matching something already shown this conversation, then fall back to whole dataset
        for movie in session.get("last_batch", []):
            if movie["title"].lower() == title.lower():
                return movie
        _, movie = find_movie_by_title(movies, title)
        if not movie:
            return {"error": f"Couldn't find '{title}' in the dataset."}
        return movie


    def log_feedback(title, watched=None, liked=None, rating=None):
        movies = session["movies"]
        movie = None
        for m in session.get("last_batch", []):
            if m["title"].lower() == title.lower():
                movie = m
                break
        if movie is None:
            _, movie = find_movie_by_title(movies, title)
        if movie is None:
            return {"error": f"Couldn't find '{title}' to log feedback for."}
        record_feedback(session["user_id"], movie["id"], movie["title"], watched=watched, rating=rating, liked=liked)
        return {"status": "logged", "title": movie["title"]}

    return {
        "get_recommendations": get_recommendations,
        "more_recommendations": more_recommendations,
        "get_movie_details": get_movie_details,
        "log_feedback": log_feedback,
    }


def _take_next_batch(session, batch_size=5):
    """Iterate through full ranked list of movies for next recommendations."""
    scored = session["last_scored"]
    offset = session["last_offset"]
    batch = scored[offset:offset + batch_size]

    if not batch:
        return {"movies": [], "note": "No more recommendations left for these filters."}

    target = session.get("last_target")
    for score, movie in batch:
        log_recommendation(session["user_id"], target, movie, score)

    session["last_batch"] = [movie for _, movie in batch]
    session["last_offset"] = offset + batch_size

    return {
        "movies": [
            {"title": m["title"], "year": m["release_date"][:4] if m["release_date"] else "N/A",
             "rating": m["vote_average"], "score": s, "overview": m["overview"]}
            for s, m in batch
        ]
    }