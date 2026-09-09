"""
Produces cached artifact (data/cf_model.npz) that later recommendation system loads. Run only once, and re-run only when want to
retrain, like whenever there is a new MovieLens release or after tuning hyperparameters.

Run it directly:
    python build_cf_model.py
"""

# Import CF building blocks
from collaborative_filtering import (
    load_ratings,
    filter_sparse_users_items,
    build_sparse_matrix,
    to_implicit_confidence,
    train_matrix_factorization,
    save_cf_model,
)

if __name__ == "__main__":
    df = load_ratings()
    df = filter_sparse_users_items(df)
    matrix, user_to_idx, item_to_idx = build_sparse_matrix(df)
    confidence_matrix = to_implicit_confidence(matrix)
    model = train_matrix_factorization(confidence_matrix)
    save_cf_model(model, user_to_idx, item_to_idx)

    print(f"\nFinished building model. {model.item_factors.shape[0]:,} movies and {model.user_factors.shape[0]:,} users each now have "
          f"a {model.item_factors.shape[1]}-dimensional CF embedding.")