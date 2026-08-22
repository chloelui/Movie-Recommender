from sentence_transformers import SentenceTransformer
import numpy as np

_model = None                                               # Module-level variable so model only loads once per run (not once per movie)

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_texts(texts):
    model = get_model()
    return model.encode(texts, show_progress_bar=True)

# Compute cosine similarity
def cosine_similarity(vec_a, vec_b):
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))