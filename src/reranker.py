from sentence_transformers import CrossEncoder
from typing import List, Dict

RERANKER_MODEL = "BAAI/bge-reranker-base"


def rerank(query: str, candidates: List[Dict], top_n: int = 5) -> List[Dict]:
    """
    Reranks a shortlist of candidates using a cross-encoder, which scores
    (query, chunk) pairs jointly for higher accuracy than embedding
    similarity alone. Only run on a small shortlist — too slow for
    full-corpus search.
    """
    model = CrossEncoder(RERANKER_MODEL)

    pairs = [(query, c["chunk"]) for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    for rank, c in enumerate(reranked):
        c["rank"] = rank + 1

    return reranked[:top_n]


if __name__ == "__main__":
    from hybrid_retriever import hybrid_search

    test_queries = [
        "what is quality diversity in neural architecture search",
        "RZ-NAS reflective zero-cost strategy",
        "RegNet design space",
    ]

    for query in test_queries:
        print(f"\n{'='*80}\nQuery: {query}\n{'='*80}")

        # Get a wider shortlist from hybrid fusion, then rerank down to top 5
        candidates = hybrid_search(query, k=10, final_k=10)
        reranked = rerank(query, candidates, top_n=5)

        for r in reranked:
            print(f"\n[Rank {r['rank']}] (rerank_score={r['rerank_score']:.4f}) Source: {r['source']}")
            print(r['chunk'][:250])