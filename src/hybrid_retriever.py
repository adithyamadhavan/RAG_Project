from pathlib import Path
from typing import List, Dict
import pickle

from retrieve import load_index_data, search as dense_search
from bm25_retriever import bm25_search
# def _get_text(r: Dict) -> str:
#     """Handles inconsistent key naming across retrieval methods (chunk vs text)."""
#     return r.get("text") or r.get("chunk") or ""


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    rrf_k: int = 60,
) -> List[Dict]:
    scores: Dict[str, float] = {}
    chunk_lookup: Dict[str, Dict] = {}

    for results in [dense_results, bm25_results]:
        for r in results:
            # chunk_text = _get_text(r)
            key = f"{r['source']}::{r['chunk'][:100]}"
            rrf_contribution = 1 / (rrf_k + r["rank"])
            scores[key] = scores.get(key, 0) + rrf_contribution
            chunk_lookup[key] = r

    ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    fused_results = []
    for rank, key in enumerate(ranked_keys):
        chunk = chunk_lookup[key]
        fused_results.append({
            "rank": rank + 1,
            "rrf_score": scores[key],
            "source": chunk["source"],
            "chunk": chunk['chunk'],
        })
    return fused_results


def hybrid_search(query: str, k: int = 10, final_k: int = 5) -> List[Dict]:
    """
    Runs dense + BM25 retrieval in parallel, fuses with RRF,
    returns the top final_k fused results.
    """
    # Dense retrieval
    index, chunks, sources, model = load_index_data()
    dense_results = dense_search(query, index, chunks, sources, model, top_k=k)

    # BM25 retrieval
    with open("bm25_data.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    bm25_results = bm25_search(
        query, bm25_data["bm25"], bm25_data["chunks"], bm25_data["sources"], top_k=k
    )

    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    return fused[:final_k]


if __name__ == "__main__":
    test_queries = [
        "what is quality diversity in neural architecture search",
        "RZ-NAS reflective zero-cost strategy",
        "RegNet design space",
    ]

    for query in test_queries:
        print(f"\n{'='*80}\nQuery: {query}\n{'='*80}")
        results = hybrid_search(query, k=10, final_k=5)
        for r in results:
            print(f"\n[Rank {r['rank']}] (rrf_score={r['rrf_score']:.4f}) Source: {r['source']}")
            print(r['chunk'][:250])