"""
Comprehensive hybrid recommender that combines semantic similarity, metadata overlap, collaborative filtering, a user's watch/like
history, and a user's current mood into a single recommendation system that provides one ranked list for the user.

Since MovieLens doesn't have data about app's users, use technique of "folding in" new user by building stand-in profile vector
for app user by averaging item factor vectors of movies user has indicated they liked via record_feedback, weighted toward likes, 
away from dislikes. Averaged vector lives in same latent space ALS learned, so can be compared against any movie's item vector with 
plain dot product. Brand new user w/ no feedback logged has no vector to build, so simply drop CF component's weight.
"""

import json
import numpy as np

from embeddings import cosine_similarity
from recommender_engine import genre_similarity, passes_filters

CF_MODEL_PATH = "data/cf_model.npz"
TMDB_TO_ML_PATH = "data/movielens_to_tmdb.json"

# Vocab translating free-text mood into genres to favor w/o hard filter
MOOD_GENRE_BOOST = {
    "cozy": {"Romance", "Family", "Comedy"},
    "feel-good": {"Comedy", "Family", "Animation"},
    "intense": {"Thriller", "Action", "Crime"},
    "scary": {"Horror", "Mystery"},
    "sad": {"Drama"},
    "mind-bending": {"Science Fiction", "Mystery", "Thriller"},
    "light": {"Comedy", "Animation", "Family"},
}

# Initial guess of how much each signal counts toward final score
DEFAULT_WEIGHTS = {
    "semantic": 0.30,
    "metadata": 0.15,
    "collaborative": 0.30,
    "personal_history": 0.15,
    "mood": 0.10,
}


class CFModel:
    """
    Wrapper around cf_model.npz file, plus TMDB <-> MovieLens id mappings. Load this once per process.
    """
    def __init__(self, model_path=CF_MODEL_PATH, mapping_path=TMDB_TO_ML_PATH):
        data = np.load(model_path, allow_pickle=True)
        self.item_factors = data["item_factors"]
        self.user_factors = data["user_factors"]

        ml_item_ids = data["item_ids"]
        self.ml_id_to_row = {str(mid): i for i, mid in enumerate(ml_item_ids)}

        with open(mapping_path, encoding="utf-8") as f:
            self.tmdb_to_ml = json.load(f)

    def item_vector(self, tmdb_id):
        """
        Looks up movie's CF embedding by TMDB id. Returns None if movie isn't in MovieLens or didn't pass MIN_RATINGS_PER_ITEM filter.
        """
        ml_id = self.tmdb_to_ml.get(str(tmdb_id))
        if ml_id is None:                               # movie doesn't exist in MovieLens
            return None
        row = self.ml_id_to_row.get(str(ml_id))
        if row is None:                                 # not enough ratings for this movie
            return None
        return self.item_factors[row]


def build_user_cf_profile(cf_model, liked_movie_ids, disliked_movie_ids=None):
    """
    Builds stand-in profile vector for app user from movies they've liked/disliked, in same latent space as trained model.

    liked_movie_ids / disliked_movie_ids are TMDB ids.

    Returns None if no usable signal (brand new user, or none of their rated movies exist in CF model). Callers must check for None and 
    skip CF component rather than setting score to 0. None profile is different from real profile that happens to score candidate at 0.
    """
    disliked_movie_ids = disliked_movie_ids or []
    vectors, weights = [], []

    for mid in liked_movie_ids:
        v = cf_model.item_vector(mid)
        if v is not None:
            vectors.append(v)
            weights.append(1.0)

    for mid in disliked_movie_ids:
        v = cf_model.item_vector(mid)
        if v is not None:
            vectors.append(-v)                          # dislikes push profile away from that movie's direction in latent space
            weights.append(0.5)                         # dislikes count for less than a like

    if not vectors:
        return None

    vectors = np.array(vectors)
    weights = np.array(weights).reshape(-1, 1)
    profile = (vectors * weights).sum(axis=0) / weights.sum()
    return profile


def collaborative_score(cf_model, user_profile, candidate_tmdb_id):
    """
    Calculate predicted affinity as dot product of user's profile vector and candidate movie's item vector. Returns None (not 0.0) 
    whenever there isn't enough data to compute real number, so tcaller can distinguish no signal from neutral signal.
    """
    if user_profile is None:
        return None
    item_vec = cf_model.item_vector(candidate_tmdb_id)
    if item_vec is None:
        return None
    return float(np.dot(user_profile, item_vec))


def build_genre_affinity(feedback_rows):
    """
    Turns user's own logged feedback into {genre_lower: score} map for personal history component. Purely about what genres user has 
    rated well before.

    feedback_rows: iterable of (genres_pipe_string, liked, rating) from user_movie_interactions

    A genre gets +1 for each liked=True occurrence, -1 for liked=False, and extra nudge toward/away based on numeric rating when present.
    """
    affinity = {}
    for genres_str, liked, rating in feedback_rows:
        if not genres_str:
            continue
        genres = {g.lower() for g in genres_str.split("|")}

        signal = 0.0
        if liked is True:
            signal += 1.0
        elif liked is False:
            signal -= 1.0
        if rating is not None:
            signal += (rating - 5.0) / 5.0              # centers 0-10 rating around 0

        for g in genres:
            affinity[g] = affinity.get(g, 0.0) + signal

    return affinity


def personal_history_score(movie, genre_affinity):
    if not genre_affinity:
        return 0.0
    movie_genres = {g.lower() for g in movie["genres"].split("|")} if movie["genres"] else set()
    scores = [genre_affinity[g] for g in movie_genres if g in genre_affinity]
    return sum(scores) / len(scores) if scores else 0.0


def mood_score(movie, mood):
    if not mood:
        return 0.0
    boosted_genres = MOOD_GENRE_BOOST.get(mood.lower())
    if not boosted_genres:
        return 0.0
    movie_genres = set(movie["genres"].split("|")) if movie["genres"] else set()
    return float(len(movie_genres & boosted_genres))


def _minmax_normalize(values):
    """
    Min-max normalize each component of hybrid model to 0-1 within current candidate pool b/c each component has different scale.
    """
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def rank_hybrid(movies,embeddings,target_index,target,filters,seen_ids,disliked_ids,
                cf_model=None,user_profile=None,genre_affinity=None,mood=None,weights=None,):
    """
    Considers user's filter requests and computes movie's score based on five independently-computed, independently-normalized 
    components combined by weighted sum.

    cf_model / user_profile / genre_affinity / mood are all optional and set to None by default. When whole component is unavailable, 
    its weight is redistributed proportionally across remaining components rather than pulling every score toward zero.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
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
    if not candidates:
        return []

    semantic_raw, metadata_raw, cf_raw, history_raw, mood_raw = [], [], [], [], []
    any_cf_signal = False

    for i, movie in candidates:
        if target is not None:
            semantic_raw.append(cosine_similarity(embeddings[target_index], embeddings[i]))
            metadata_raw.append(genre_similarity(target, movie))
        else:
            semantic_raw.append(0.0)
            movie_genres = {g.lower() for g in movie["genres"].split("|")} if movie["genres"] else set()
            metadata_raw.append(len(movie_genres & include_genres))

        cf_val = None
        if cf_model is not None and user_profile is not None:
            cf_val = collaborative_score(cf_model, user_profile, movie["id"])
        if cf_val is not None:
            any_cf_signal = True
        cf_raw.append(cf_val if cf_val is not None else 0.0)

        history_raw.append(personal_history_score(movie, genre_affinity))
        mood_raw.append(mood_score(movie, mood))

    semantic_n = _minmax_normalize(semantic_raw)
    metadata_n = _minmax_normalize(metadata_raw)
    cf_n = _minmax_normalize(cf_raw) if any_cf_signal else np.zeros(len(candidates))
    history_n = _minmax_normalize(history_raw)
    mood_n = _minmax_normalize(mood_raw)

    # Redistribute weight away from any component w/ nothing to contribute this turn
    active_weights = dict(weights)
    if not any_cf_signal:
        active_weights["collaborative"] = 0.0
    if not genre_affinity:
        active_weights["personal_history"] = 0.0
    if not mood:
        active_weights["mood"] = 0.0

    total = sum(active_weights.values())
    if total <= 0:
        active_weights = weights                        # if everything was missing, just use configured weights as-is
        total = sum(active_weights.values())
    active_weights = {k: v / total for k, v in active_weights.items()}

    scored = []
    for idx, (i, movie) in enumerate(candidates):
        score = (
            active_weights.get("semantic", 0) * semantic_n[idx]
            + active_weights.get("metadata", 0) * metadata_n[idx]
            + active_weights.get("collaborative", 0) * cf_n[idx]
            + active_weights.get("personal_history", 0) * history_n[idx]
            + active_weights.get("mood", 0) * mood_n[idx]
        )
        scored.append((round(float(score), 4), movie))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored