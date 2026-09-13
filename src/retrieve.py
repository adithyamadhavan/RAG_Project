import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

Model_name = "BAAI/bge-small-en-v1.5"

def load_index_data():
    index = faiss.read_index("faiss_index.bin")
    with open("chunk_data.pkl", "rb") as f:
        data = pickle.load(f)
    model = SentenceTransformer(Model_name)
    return index, data['chunks'], data['sources'], model


def search(query: str, index, chunks: list, sources: list, model: SentenceTransformer, top_k: int = 5):
    """
    Searches the FAISS index for the most similar chunks to the given query.
    Args:
        query (str): The search query.
        index: The FAISS index.
        chunks (list): The list of text chunks.
        sources (list): The list of source filenames corresponding to each chunk.
        model (SentenceTransformer): The embedding model used to encode the query.
        top_k (int): The number of top results to return. Default is 5.
    """

    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "chunk": chunks[idx],
            "source": sources[idx],
            "distance": float(distances[0][rank])
        })
    return results

if __name__ == "__main__":
    index, chunks ,sources,model =load_index_data()
    test_query = ["what is quality diversity in neural architecture search",
        "how does CoLLM-NAS coordinate between multiple LLMs",
        "what is RegNet's design space",
        "how does CoLLM-NAS coordinate between multiple LLMs",
        "what is the difference between CoLLM-NAS and other NAS methods"]
    for query in test_query:
        print(f"Query:{query}")

        results = search(query, index, chunks, sources, model,top_k=5)
        for res in results:
            print(f"Rank: {res['rank']}, Source: {res['source']}, Distance: {res['distance']:.4f}")
            print(f"Chunk: {res['chunk']}\n")
           