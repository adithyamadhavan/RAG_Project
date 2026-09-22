import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_contextual_index():
    index = faiss.read_index("faiss_index_contextual.bin")
    with open("chunk_data_contextual.pkl", "rb") as f:
        data = pickle.load(f)
    model = SentenceTransformer(MODEL_NAME)
    return index, data["chunks"], data["sources"], model


def search(query: str, index, chunks, sources, model, k: int = 5):
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, k)
    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "distance": float(distances[0][rank]),
            "source": sources[idx],
            "chunk": chunks[idx],
        })
    return results


if __name__ == "__main__":
    index, chunks, sources, model = load_contextual_index()

    test_queries = [
        "what is quality diversity in neural architecture search",
        "RZ-NAS reflective zero-cost strategy",
        "RegNet design space",
        "how would CoLLM-NAS's Coordinator module apply to a completely different domain outside NAS",
    ]

    for query in test_queries:
        print(f"\n{'='*80}\nQuery: {query}\n{'='*80}")
        results = search(query, index, chunks, sources, model, k=5)
        for r in results:
            print(f"\n[Rank {r['rank']}] (distance={r['distance']:.4f}) Source: {r['source']}")
            print(r['chunk'][:250])