from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
from typing import List, Dict

def build_bm25_index(chunks: List[str]) -> BM25Okapi:
    """
    Builds a BM25 index over tokenized chunks.
    BM25 uses term-frequency scoring, not embeddings — good for exact
    keyword/acronym matches (e.g. 'RZ-NAS', 'CodeGen-Mono-350M') that
    dense embeddings can sometimes blur.
    """
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks)

def bm25_search(query: str, bm25: BM25Okapi, chunks: List[str], sources: List[str], top_k: int = 5) -> List[Dict]:
    """Returns top-k chunks ranked by BM25 score for the given query."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Get indices of top-k scores, descending
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        results.append({
            "rank": rank + 1,
            "score": float(scores[idx]),
            "source": sources[idx],
            "text": chunks[idx],
            "chunk_index": idx,  # needed later for RRF fusion
        })
    return results

if __name__ == "__main__":
    from ingest import ingest_pdf_files
    from chunkers_recursive import chunk_corpus

    corpus = ingest_pdf_files(Path("corpus"))
    chunked = chunk_corpus(corpus)

    all_chunks: List[str] = []
    chunk_sources: List[str] = []
    for filename, chunks in chunked.items():
        all_chunks.extend(chunks)
        chunk_sources.extend([filename] * len(chunks))

    print(f"Building BM25 index over {len(all_chunks)} chunks...")
    bm25 = build_bm25_index(all_chunks)

    # Save for reuse
    with open("bm25_data.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": all_chunks, "sources": chunk_sources}, f)

    print("BM25 index built and saved.")

    # Quick comparison test — same queries as your dense retrieval test
    test_queries = [
        "what is quality diversity in neural architecture search",
        "RZ-NAS reflective zero-cost strategy",
        "RegNet design space",
    ]

    for query in test_queries:
        print(f"\n{'='*80}\nQuery: {query}\n{'='*80}")
        results = bm25_search(query, bm25, all_chunks, chunk_sources, top_k=3)
        for r in results:
            print(f"\n[Rank {r['rank']}] (score={r['score']:.3f}) Source: {r['source']}")
            print(r['text'][:250])