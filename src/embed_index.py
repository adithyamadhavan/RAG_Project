import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer


Model_name = "BAAI/bge-small-en-v1.5"

def build_faiss_index(chunks: List[str], model_name: str = Model_name) -> Tuple[FAISS, SentenceTransformer]:
    """
    Builds a FAISS index from a list of text chunks using the specified embedding model.
    Args:
        chunks (List[str]): A list of text chunks to be indexed.
        model_name (str): The name of the embedding model to use. Default is "BAAI/bge-small-en-v1.5".
    Returns:
        Tuple[FAISS, SentenceTransformer]: A tuple containing the FAISS index and the embedding model.
    """
    # Load the embedding model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings for each chunk
    embeddings = model.encode(chunks, convert_to_numpy=True,show_progress_bar=True)
    
    # Create a FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to the index
    index.add(embeddings)
    
    # Wrap the FAISS index in a LangChain FAISS vector store
   # faiss_index = FAISS(index=index, embedding_function=model.encode)
    
    return index, embeddings



if __name__ == "__main__":
    from ingest import ingest_pdf_files
    from chunkers_recursive import chunk_corpus

    # Ingest PDF files from the "corpus" directory
    corpus = ingest_pdf_files(Path("corpus"))
    
    # Chunk the corpus using recursive chunking
    chunked_corpus = chunk_corpus(corpus)
    all_chunks: List[str] = []
    chunk_sources: List[str] = []
    for filename, chunks in chunked_corpus.items():
        all_chunks.extend(chunks)
        chunk_sources.extend([filename] * len(chunks))

    print(f"Total chunks across corpus: {len(all_chunks)}")

    index, embeddings = build_faiss_index(all_chunks)

    print(f"FAISS index built. Total vectors: {index.ntotal}, dimension: {embeddings.shape[1]}")

    # Save for reuse in the next step (retrieval testing)
    faiss.write_index(index, "faiss_index.bin")
    import pickle
    with open("chunk_data.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "sources": chunk_sources}, f)

    print("Saved index and chunk metadata to disk.")