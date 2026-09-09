"""
Creates a library of building blocks that build_cf_model.py calls in order. Collaborative-filtering model can be re-trained later 
without duplicating logic, and other files can import individual pieces.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

RATINGS_PATH = "data/movielens/ratings.csv"

# Keep only those w/ enough ratings to generalize patterns
MIN_RATINGS_PER_USER = 5
MIN_RATINGS_PER_ITEM = 10

def load_ratings(path=RATINGS_PATH):
    dtypes = {"userId": "int32", "movieId": "int32", "rating": "float32"}
    print("Loading ratings.csv...")
    df = pd.read_csv(path, dtype=dtypes, usecols=["userId", "movieId", "rating"])
    print(f"Loaded {len(df):,} ratings from {df['userId'].nunique():,} users across {df['movieId'].nunique():,} movies.")
    return df

def filter_sparse_user_items(df, min_user=MIN_RATINGS_PER_USER, min_item=MIN_RATINGS_PER_ITEM):
    """
    Drop users and movies w/ too little data to be useful. Filter items first, then users, rather than both at once to keep two
    constraints from fighting each other.
    """
    before = len(df)

    item_counts = df["movieId"].value_counts()
    keep_items = item_counts[item_counts >= min_item].index
    df = df[df["movieId"].isin(keep_items)]

    user_counts = df["userId"].value_counts()
    keep_users = user_counts[user_counts >= min_user].index
    df = df[df["userId"].isin(keep_users)]

    print(f"Filtered {before:,} -> {len(df):,} ratings (kept users w/ >={min_user} ratings, movies w/ >={min_item} ratings).")
    return df

def build_sparse_matrix(df):
    """
    Reshapes (userId, movieId, rating) table into sparse matrix where row = user, column = movie, value = rating. Dense version 
    would be ~16 billion cells, sparse matrix only stores the cells that actually have value.

    MovieLens ids aren't contiguous, so can't use them directly as matrix positions. Build own 0 to n-1 index for both users and movies 
    and remember mapping b/c will need to translate back and forth b/w movieId and row/column number for rest of pipeline.
    """
    user_ids = df["userId"].unique()
    item_ids = df["movieId"].unique()

    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {m: i for i, m in enumerate(item_ids)}

    rows = df["userId"].map(user_to_idx).to_numpy()
    cols = df["movieId"].map(item_to_idx).to_numpy()
    vals = df["rating"].to_numpy()

    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    print(f"Built a {matrix.shape[0]:,} x {matrix.shape[1]:,} sparse matrix "
          f"({matrix.nnz:,} non-zero entries, {matrix.nnz / (matrix.shape[0]*matrix.shape[1]) * 100:.4f}% dense).")
    return matrix, user_to_idx, item_to_idx

def to_implicit_confidence(matrix, alpha=15.0):
    """
    Use implicit feedback to treat fact that rating exists at all as signal of a user engaged w/ movie enough to watch and rate it. 
    Actual star rating value becomes confidencemultiplier on that signal rather than a preference score.
    """
    confidence_matrix = matrix.copy()
    # Every movie gets baseline confidence 1 and more confidence for higher rating
    confidence_matrix.data = 1.0 + alpha * confidence_matrix.data                       # standard formula: confidence = 1 + alpha * rating
    return confidence_matrix

def train_matrix_factorization(confidence_matrix, factors=64, regularization=0.1, iterations=20):
    """
    Experiment of matrix factorization + embeddings to generate (user_row . movie_row) dot products that represent predicted 
    affinity score for any movie, including ones that user never rated.
    """
    import implicit

    model = implicit.als.AlternatingLeastSquares(factors=factors,regularization=regularization,iterations=iterations,random_state=42,)
    print(f"Training ALS: factors={factors}, regularization={regularization}, iterations={iterations}...")
    model.fit(confidence_matrix)
    print("Training complete.")
    return model

def item_based_knn(confidence_matrix, k=10):
    """
    Experiment for KNN recommendations to test against factorization method. Represents each movie as column of ratings across every user,
    then find movies w/ close vectors by cosine similarity. Two movies rated highly by mostly same people end up near each other.

    Note: scales worse than compressed factor vectors ALS and doesn't naturally give per-user profile like factorization does.
    """
    item_vectors = confidence_matrix.T.tocsr()                                          # rows become movies now
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k + 1)
    nn.fit(item_vectors)
    return nn, item_vectors

def nearest_movies(nn, item_vectors, item_idx, item_to_idx, k=10):
    """Look up k movies most similar to item_idx according to fitted item_based_knn model."""
    idx_to_item = {v: k_ for k_, v in item_to_idx.items()}
    distances, indices = nn.kneighbors(item_vectors[item_idx], n_neighbors=k + 1)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == item_idx:
            continue  # a movie is always its own closest neighbor, skip it
        results.append((idx_to_item[idx], 1 - dist))  # cosine distance -> similarity
    return results[:k]

def save_cf_model(model, user_to_idx, item_to_idx, path="data/cf_model.npz"):
    """
    Saves everything needed later to make recommendations:
      - item_factors/user_factors: learned embedding matrices
      - item_ids/user_ids: MovieLens ids in same order as factor rows, so you can map factor row back to movieId/userId
    """
    np.savez(
        path,
        item_factors=model.item_factors,
        user_factors=model.user_factors,
        item_ids=np.array(list(item_to_idx.keys())),
        user_ids=np.array(list(user_to_idx.keys())),
    )
    print(f"Saved CF model to {path}")