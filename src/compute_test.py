import json
from eval_metrics import mean_metric, reciprocal_rank, ndcg


def labels_to_relevance_lists(labels_dict, results_path):
    """
    Converts {question: [relevant_chunk_numbers]} into
    a list of 0/1 relevance lists in ranked order, matching
    the order questions appear in the results file.
    """
    with open(results_path) as f:
        results = json.load(f)

    relevance_lists = []
    for r in results:
        query = r["query"]
        if query not in labels_dict:
            continue
        relevant_ranks = set(labels_dict[query])  # e.g. {1, 3}
        num_chunks = len(r["retrieved_chunks"])
        # build a 0/1 list: 1 if that rank (1-indexed) was marked relevant
        relevance = [1 if (i + 1) in relevant_ranks else 0 for i in range(num_chunks)]
        relevance_lists.append(relevance)

    return relevance_lists


if __name__ == "__main__":
    with open("relevance_labels.json") as f:
        labels = json.load(f)

    naive_relevance = labels_to_relevance_lists(labels["naive"], "eval_results_naive.json")
    hybrid_relevance = labels_to_relevance_lists(labels["hybrid"], "eval_results_hybrid.json")

    print(f"Naive:  {len(naive_relevance)} questions scored")
    print(f"Hybrid: {len(hybrid_relevance)} questions scored")

    naive_mrr = mean_metric(naive_relevance, reciprocal_rank)
    naive_ndcg = mean_metric(naive_relevance, ndcg)

    hybrid_mrr = mean_metric(hybrid_relevance, reciprocal_rank)
    hybrid_ndcg = mean_metric(hybrid_relevance, ndcg)

    print(f"\n{'='*50}")
    print(f"{'Metric':<15}{'Naive':<15}{'Hybrid+Rerank':<15}")
    print(f"{'='*50}")
    print(f"{'MRR':<15}{naive_mrr:<15.4f}{hybrid_mrr:<15.4f}")
    print(f"{'NDCG@5':<15}{naive_ndcg:<15.4f}{hybrid_ndcg:<15.4f}")