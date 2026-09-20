import json

with open("relevance_labels.json") as f:
    labels = json.load(f)

with open("eval_results_naive.json") as f:
    naive_results = json.load(f)

with open("eval_results_hybrid.json") as f:
    hybrid_results = json.load(f)


def first_relevant_rank(labels_list):
    """Returns the rank (1-indexed) of the first relevant chunk, or None."""
    return min(labels_list) if labels_list else None


print(f"{'Question':<70}{'Naive rank':<12}{'Hybrid rank':<12}{'Change'}")
print("=" * 110)

regressions = []
improvements = []

for n, h in zip(naive_results, hybrid_results):
    query = n["query"]
    naive_first = first_relevant_rank(labels["naive"].get(query, []))
    hybrid_first = first_relevant_rank(labels["hybrid"].get(query, []))

    n_str = str(naive_first) if naive_first else "none"
    h_str = str(hybrid_first) if hybrid_first else "none"

    change = ""
    if naive_first and hybrid_first:
        if hybrid_first > naive_first:
            change = f"WORSE (+{hybrid_first - naive_first})"
            regressions.append((query, naive_first, hybrid_first))
        elif hybrid_first < naive_first:
            change = f"BETTER (-{naive_first - hybrid_first})"
            improvements.append((query, naive_first, hybrid_first))
        else:
            change = "same"
    elif naive_first and not hybrid_first:
        change = "WORSE (lost relevant chunk entirely)"
        regressions.append((query, naive_first, "none"))
    elif hybrid_first and not naive_first:
        change = "BETTER (gained relevant chunk)"
        improvements.append((query, "none", hybrid_first))

    print(f"{query[:68]:<70}{n_str:<12}{h_str:<12}{change}")

print(f"\n\nRegressions: {len(regressions)}")
for q, n, h in regressions:
    print(f"  {q[:70]} — naive rank {n} -> hybrid rank {h}")

print(f"\nImprovements: {len(improvements)}")
for q, n, h in improvements:
    print(f"  {q[:70]} — naive rank {n} -> hybrid rank {h}")