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