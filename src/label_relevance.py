import json
import os

def load_results(path):
    with open(path) as f:
        return json.load(f)


def parse_input(raw: str):
    """Safely parses '1,3' / 'none' / '' into a list of ints."""
    raw = raw.strip()
    if raw == "" or raw.lower() == "none":
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def label_interactively(naive_path, hybrid_path, output_path):
    naive_results = load_results(naive_path)
    hybrid_results = load_results(hybrid_path)

    # Resume support: load existing labels if the file already exists
    if os.path.exists(output_path):
        with open(output_path) as f:
            labels = json.load(f)
        print(f"Resuming — {len(labels['naive'])} questions already labeled.")
    else:
        labels = {"naive": {}, "hybrid": {}}

    for i, (n, h) in enumerate(zip(naive_results, hybrid_results)):
        if n["query"] in labels["naive"] and h["query"] in labels["hybrid"]:
            continue  # already labeled, skip

        print(f"\n{'='*90}\n[{i+1}/{len(naive_results)}] {n['query']}\n{'='*90}")

        print("\n--- NAIVE retrieved chunks ---")
        for j, chunk in enumerate(n["retrieved_chunks"]):
            print(f"\n  [{j+1}] {chunk[:200]}")
        naive_raw = input("\nNaive: relevant chunk numbers (e.g. '1,3', 'none', blank=none): ")
        labels["naive"][n["query"]] = parse_input(naive_raw)

        print("\n--- HYBRID+RERANK retrieved chunks ---")
        for j, chunk in enumerate(h["retrieved_chunks"]):
            print(f"\n  [{j+1}] {chunk[:200]}")
        hybrid_raw = input("\nHybrid: relevant chunk numbers (e.g. '1,3', 'none', blank=none): ")
        labels["hybrid"][h["query"]] = parse_input(hybrid_raw)

        # Save after EVERY question — no more losing progress on crash
        with open(output_path, "w") as f:
            json.dump(labels, f, indent=2)

    print(f"\nAll done. Saved labels to {output_path}")


if __name__ == "__main__":
    label_interactively("eval_results_naive.json", "eval_results_hybrid.json", "relevance_labels.json")