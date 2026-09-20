import numpy as np
from typing import List


def reciprocal_rank(relevance: List[int]) -> float:
    """
    relevance: list of 0/1 labels in ranked order (1 = relevant chunk).
    Returns 1/rank of the first relevant hit, or 0 if none found.
    """
    for i, rel in enumerate(relevance):
        if rel == 1:
            return 1 / (i + 1)
    return 0.0


def dcg(relevance: List[int]) -> float:
    """Discounted Cumulative Gain — rewards relevant hits near the top."""
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))


def ndcg(relevance: List[int]) -> float:
    """
    Normalized DCG — divides by the ideal DCG (all relevant items
    sorted to the top) so scores are comparable across queries.
    """
    ideal_relevance = sorted(relevance, reverse=True)
    ideal = dcg(ideal_relevance)
    if ideal == 0:
        return 0.0
    return dcg(relevance) / ideal


def mean_metric(all_relevance_lists: List[List[int]], metric_fn) -> float:
    """Averages a per-query metric across all queries."""
    scores = [metric_fn(rel) for rel in all_relevance_lists]
    return sum(scores) / len(scores)


if __name__ == "__main__":
    # Example: 3 queries, each with relevance labels for top-5 retrieved chunks
    example_relevance = [
        [0, 1, 0, 0, 1],  # relevant chunk at rank 2 and 5
        [1, 0, 0, 0, 0],  # relevant chunk at rank 1
        [0, 0, 0, 0, 0],  # no relevant chunk found in top-5
    ]

    mrr = mean_metric(example_relevance, reciprocal_rank)
    ndcg_score = mean_metric(example_relevance, ndcg)

    print(f"MRR: {mrr:.4f}")
    print(f"NDCG@5: {ndcg_score:.4f}")