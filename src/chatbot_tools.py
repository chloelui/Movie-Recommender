from db import log_recommendation, record_feedback, record_detail_view
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
    """Merges this turn's fields into accumulated filters. Returns (active, conflicts), so caller can tell the user what happened 
    rather than silently dropping one side of contradiction."""
    conflicts = []

    for key in ("include_genres", "exclude_genres", "include_actors", "exclude_actors"):
        if key in updates and updates[key]:
            active[key] |= {v.lower() for v in updates[key]}

    # Resolve genre contradictions to keep most recent one
    genre_overlap = active["include_genres"] & active["exclude_genres"]
    if genre_overlap:
        include_updated_now = bool(updates.get("include_genres"))
        exclude_updated_now = bool(updates.get("exclude_genres"))

        if exclude_updated_now and not include_updated_now:
            # user just excluded something they previously included
            active["include_genres"] -= genre_overlap
            resolution = "excluded"
        else:
            # newly included wins, or both/neither updated this turn
            active["exclude_genres"] -= genre_overlap
            resolution = "included"

        conflicts.append({
            "field": "genres",
            "values": sorted(genre_overlap),
            "resolved_as": resolution,
        })

    actor_overlap = active["include_actors"] & active["exclude_actors"]
    if actor_overlap:
        include_updated_now = bool(updates.get("include_actors"))
        exclude_updated_now = bool(updates.get("exclude_actors"))

        if exclude_updated_now and not include_updated_now:
            active["include_actors"] -= actor_overlap
            resolution = "excluded"
        else:
            active["exclude_actors"] -= actor_overlap
            resolution = "included"

        conflicts.append({
            "field": "actors",
            "values": sorted(actor_overlap),
            "resolved_as": resolution,
        })

    for key in ("min_year", "max_year", "min_rating"):
        if key in updates and updates[key] is not None:
            active[key] = updates[key]

    # min_year after max_year or min_rating that can't be satisfied
    if active["min_year"] is not None and active["max_year"] is not None and active["min_year"] > active["max_year"]:
        conflicts.append({
            "field": "year_range",
            "values": [active["min_year"], active["max_year"]],
            "resolved_as": "unresolved",  # left as-is to flag to user
        })

    if "target_movie" in updates and updates["target_movie"]:
        active["target_movie"] = updates["target_movie"]

    return active, conflicts


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

        active, conflicts = merge_filters(session["active_filters"], kwargs)

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
        if conflicts:
            result["filter_conflicts"] = conflicts
        return result


    def more_recommendations():
        if "last_scored" not in session:
            return {"error": "No previous recommendations to continue from."}
        return _take_next_batch(session)


    def get_movie_details(title):
        movies = session["movies"]
        # Try something already shown in convo then fall back to all movies
        for movie in session.get("last_batch", []):
            if movie["title"].lower() == title.lower():
                record_detail_view(session["user_id"], movie["id"], movie["title"])
                result = dict(movie)
                result["title_match"] = {"status": "exact", "queried_title": title, "matched_title": movie["title"]}
                return result

        match = find_movie_by_title(movies, title)
        if not match["movie"]:
            return {
                "error": f"Couldn't find '{title}' in the dataset.",
                "title_match": {"status": match["status"], "queried_title": title, "matched_title": None},
            }

        record_detail_view(session["user_id"], match["movie"]["id"], match["movie"]["title"])
        result = dict(match["movie"])
        result["title_match"] = {
            "status": match["status"],
            "queried_title": match["queried_title"],
            "matched_title": match["matched_title"],
        }
        return result


    def log_feedback(title, watched=None, liked=None, rating=None):
        movies = session["movies"]
        movie = None
        match_status = "exact"
        matched_title = None

        for m in session.get("last_batch", []):
            if m["title"].lower() == title.lower():
                movie = m
                matched_title = m["title"]
                break

        if movie is None:
            match = find_movie_by_title(movies, title)
            movie = match["movie"]
            match_status = match["status"]
            matched_title = match["matched_title"]

        if movie is None:
            return {
                "error": f"Couldn't find '{title}' to log feedback for.",
                "title_match": {"status": match_status, "queried_title": title, "matched_title": None},
            }

        # Confirm guess with user instead of logging movie they may not have meant
        if match_status == "fuzzy":
            return {
                "status": "needs_confirmation",
                "title_match": {"status": "fuzzy", "queried_title": title, "matched_title": matched_title},
                "note": "Do not log this yet. Ask the user to confirm the matched title before calling log_feedback again.",
            }

        record_feedback(session["user_id"], movie["id"], movie["title"], watched=watched, rating=rating, liked=liked)
        return {
            "status": "logged",
            "title": movie["title"],
            "title_match": {"status": match_status, "queried_title": title, "matched_title": matched_title},
        }

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