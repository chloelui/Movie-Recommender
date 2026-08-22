import csv
import numpy as np
from embeddings import embed_texts

# Precompute embeddings once
with open("data/movies.csv", "r", encoding="utf-8") as file:
    movies = list(csv.DictReader(file))

# Embed movie overviews
overviews = [movie["overview"] for movie in movies]
vectors = embed_texts(overviews)

# Cache computed embeddings to disk
np.save("data/movie_embeddings.npy", vectors)
print(f"Saved embeddings for {len(movies)} movies.")