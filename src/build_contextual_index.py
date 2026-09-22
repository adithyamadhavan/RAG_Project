from pathlib import Path
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ingest import ingest_pdf_files
from chunkers_recursive import chunk_corpus
from contextual_chunker import build_contextual_subset

MODEL_NAME = "BAAI/bge-small-en-v1.5"

if __name__ == "__main__":
    corpus = ingest_pdf_files(Path("corpus"))
    chunked = chunk_corpus(corpus)

    contextualized, original_chunks, sources = build_contextual_subset(chunked, chunks_per_doc=30)

    print("\nBuilding FAISS index over contextualized subset...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(contextualized, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, "faiss_index_contextual.bin")
    with open("chunk_data_contextual.pkl", "wb") as f:
        # store ORIGINAL chunk text (not contextualized) for display/generation,
        # but the index was built from contextualized embeddings
        pickle.dump({"chunks": original_chunks, "sources": sources}, f)

    print(f"Saved contextual index: {index.ntotal} vectors")