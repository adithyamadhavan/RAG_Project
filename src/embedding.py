from sentence_transformers import SentenceTransformer
from typing import List, Optional


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[List[float]]:
    model = get_embedding_model(model_name)
    return model.encode(texts).tolist()


def embed_query(query: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[float]:
    model = get_embedding_model(model_name)
    return model.encode(query).tolist()