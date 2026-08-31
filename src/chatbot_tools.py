from db import log_recommendation, record_feedback
from recommender_engine import find_movie_by_title, generate_recommendations, build_actor_vocab, build_genre_vocab, validate_values


def default_filters():
    """Shape of empty search to initialize a session and to wipe filters when the user explicitly starts over."""
    return {
        "target_movie": None,
        "include_genres": set(),
        "exclude_genres": set(),
        "include_actors": set(),
        "exclude_actors": set(),
        "min_year": None,
        "max_year": None,
        "min_rating": None,
    }


def merge_filters(active, updates):
    """Merges this turn's explicitly-provided fields into accumulated filter state. Only keys actually 
    present in `updates` are touched. Fields user didn't mention this turn are left same."""
    for key in ("include_genres", "exclude_genres", "include_actors", "exclude_actors"):
        if key in updates and updates[key]:
            active[key] |= {v.lower() for v in updates[key]}

    # Keep most recent filters if contradict previous filters
    if "include_genres" in updates and updates["include_genres"]:
        active["exclude_genres"] -= active["include_genres"]
    if "exclude_genres" in updates and updates["exclude_genres"]:
        active["include_genres"] -= active["exclude_genres"]
    if "include_actors" in updates and updates["include_actors"]:
        active["exclude_actors"] -= active["include_actors"]
    if "exclude_actors" in updates and updates["exclude_actors"]:
        active["include_actors"] -= active["exclude_actors"]

    for key in ("min_year", "max_year", "min_rating"):
        if key in updates and updates[key] is not None:
            active[key] = updates[key]

    if "target_movie" in updates and updates["target_movie"]:
        active["target_movie"] = updates["target_movie"]

    return active


def build_tools(session):
    """Returns the four callable actions, all closed over user's session state
    (movies, embeddings, user_id, what's been shown so far)."""
    
    # Initialize reusables when session starts
    session.setdefault("active_filters", default_filters())
    session.setdefault("genre_vocab", build_genre_vocab(session["movies"]))
    session.setdefault("actor_vocab", build_actor_vocab(session["movies"]))

    def get_recommendations(**kwargs):
        movies = session["movies"]
        embeddings = session["embeddings"]

        reset = kwargs.pop("reset", False)
        if reset:
            session["active_filters"] = default_filters()

        # Validate genre/actor values before merging them into filters
        validation_report = {}
        for field, vocab_key in [("include_genres", "genre_vocab"), ("exclude_genres", "genre_vocab"), 
                                 ("include_actors", "actor_vocab"), ("exclude_actors", "actor_vocab"),]:
            if field in kwargs and kwargs[field]:
                val = validate_values(kwargs[field], session[vocab_key])
                kwargs[field] = val["resolved"]  # only real/corrected values proceed to filtering
                if val["corrected"] or val["unmatched"]:
                    validation_report[field] = {
                        "corrected": val["corrected"],
                        "unmatched": val["unmatched"],
                    }

        active = merge_filters(session["active_filters"], kwargs)

        # Only report title match if user mentioned movie this turn
        title_mentioned_this_turn = "target_movie" in kwargs and kwargs["target_movie"]

        match = find_movie_by_title(movies, active["target_movie"]) if active["target_movie"] else \
            {"status": "no_query", "index": None, "movie": None, "matched_title": None, "queried_title": None}

        target_index, target = match["index"], match["movie"]

        if match["status"] == "fuzzy":
            target_index, target = None, None
            active["target_movie"] = None  # don't re-guess bad anchor on future turns

        scored = generate_recommendations(
            movies, embeddings, target_index, target, active,
            session["seen_ids"], session["disliked_ids"]
        )

        session["last_scored"] = scored
        session["last_offset"] = 0
        session["last_target"] = target

        result = _take_next_batch(session)
        if title_mentioned_this_turn:
            result["title_match"] = {
                "status": match["status"],
                "queried_title": match["queried_title"],
                "matched_title": match["matched_title"],
            }
        result["active_filters_summary"] = {
            "target_movie": active["target_movie"],
            "include_genres": sorted(active["include_genres"]),
            "exclude_genres": sorted(active["exclude_genres"]),
            "include_actors": sorted(active["include_actors"]),
            "exclude_actors": sorted(active["exclude_actors"]),
            "min_year": active["min_year"],
            "max_year": active["max_year"],
            "min_rating": active["min_rating"],
        }
        if validation_report:
            result["filter_validation"] = validation_report
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