# src/check_eval_set.py
# Sanity-checks the eval question set against YOUR index before you trust it.
# No LLM calls, takes a few seconds.
#
#   python src/check_eval_set.py
#
# For each qa question it retrieves the top 3 chunks (the same search the system
# uses) and reports:
#   - source:   is the expected paper among the retrieved chunks?
#   - keywords: which keywords appear in the retrieved text? A correct answer can
#               only contain words the model can see, so if fewer than min_hits
#               keywords appear in the chunks, the question is marked WEAK:
#               either change the keywords or reword the question.
# For each out-of-corpus question it prints the best retrieval distance and
# whether it would be stopped by the distance gate (MAX_DIST, default 0.8).
import json
import os

from retrieve import load_index_data, search

EVAL_PATH = os.environ.get("EVAL_SET", "evals/agent_eval_set.json")
MAX_DIST = float(os.environ.get("MAX_DIST", "0.8"))


def dist(r):
    for k in ("distance", "dist", "score"):
        if isinstance(r.get(k), (int, float)):
            return float(r[k])
    return None


def main():
    index, chunks, sources, model = load_index_data()
    items = json.load(open(EVAL_PATH))
    weak = 0
    print(f"{'id':<14}{'check':<8}detail")
    for it in items:
        res = search(it["question"], index, chunks, sources, model, 3)
        text = " ".join(r["chunk"] for r in res).lower()
        if it["type"] == "qa":
            kws = it.get("keywords") or []
            found = [k for k in kws if k.lower() in text]
            missing = [k for k in kws if k.lower() not in text]
            src = it.get("expected_source", "")
            src_ok = any(src.lower() in str(r["source"]).lower() for r in res) if src else None
            need = it.get("min_hits", max(1, len(kws) // 2))
            ok = len(found) >= need and src_ok is not False
            weak += not ok
            print(f"{it['id']:<14}{'ok' if ok else 'WEAK':<8}source_in_top3={src_ok} "
                  f"keywords {len(found)}/{len(kws)} (need {need}) missing={missing}")
        else:
            ds = [dist(r) for r in res if dist(r) is not None]
            best = min(ds) if ds else None
            gated = best is not None and best > MAX_DIST
            print(f"{it['id']:<14}{'ooc':<8}best_distance={best:.3f} "
                  f"{'stopped by distance gate' if gated else 'PASSES the gate (needs the checker)'}")
    print(f"\n{weak} weak question(s). Fix those before reading the eval numbers.")


if __name__ == "__main__":
    main()