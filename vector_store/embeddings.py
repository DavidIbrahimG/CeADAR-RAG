from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class STEmbeddingFunction:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: List[str]) -> List[List[float]]: 
        embeddings = self.model.encode(input, show_progress_bar=False)
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return [e.tolist() for e in embeddings]

def get_embedding_function():
    return STEmbeddingFunction()
