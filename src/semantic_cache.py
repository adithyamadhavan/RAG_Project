# src/semantic_cache.py
# Semantic cache: if a new question is close enough in embedding space to one
# already answered (and approved), return the stored answer with no retrieval
# and no LLM call.
#
# Rules (set in agentic_rag.py):
#   - only RAG answers that a human approved are written
#   - abstentions (NOT_IN_CORPUS), tool/math answers and blocked requests are never cached
#   - dedupe on write: a near-identical question updates its entry instead of adding one
#   - entries expire after CACHE_TTL seconds; CACHE.clear() invalidates everything
#     (call it when the index is rebuilt)
#
#   python src/semantic_cache.py calibrate   -> similarity of paraphrase vs similar-but-different pairs
#   python src/semantic_cache.py demo        -> ask / paraphrase / different question, shows hit or miss
#
# Env: USE_CACHE=1 turns it on in agentic_rag (default OFF, so benchmarks stay honest)
#      CACHE_THRESHOLD (default 0.95), CACHE_TTL seconds (default 3600)
import os
import sys
import time
from collections import Counter

import numpy as np

THRESHOLD = float(os.environ.get("CACHE_THRESHOLD", "0.95"))
TTL = float(os.environ.get("CACHE_TTL", "3600"))
MAX_ENTRIES = 500


def _embed(model, text):
    return np.asarray(model.encode(text, normalize_embeddings=True), dtype="float32")


class SemanticCache:
    def __init__(self, threshold=THRESHOLD, ttl=TTL, max_entries=MAX_ENTRIES):
        self.threshold, self.ttl, self.max_entries = threshold, ttl, max_entries
        self.entries = []          # dicts: question, emb, answer, ts
        self.stats = Counter()

    def _best(self, emb):
        now = time.time()
        self.entries = [e for e in self.entries if now - e["ts"] <= self.ttl]
        if not self.entries:
            return None, 0.0
        sims = np.stack([e["emb"] for e in self.entries]) @ emb
        i = int(np.argmax(sims))
        return i, float(sims[i])

    def lookup(self, model, question):
        """Returns (answer, similarity) on a hit, else (None, best_similarity)."""
        i, sim = self._best(_embed(model, question))
        if i is not None and sim >= self.threshold:
            self.stats["hits"] += 1
            return self.entries[i]["answer"], sim
        self.stats["misses"] += 1
        return None, sim

    def put(self, model, question, answer):
        emb = _embed(model, question)
        i, sim = self._best(emb)
        if i is not None and sim >= self.threshold:   # dedupe: refresh, don't append
            self.entries[i].update(answer=answer, ts=time.time())
            self.stats["refreshed"] += 1
            return
        self.entries.append({"question": question, "emb": emb,
                             "answer": answer, "ts": time.time()})
        self.stats["writes"] += 1
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def clear(self):
        self.entries.clear()


CACHE = SemanticCache()

# Pairs for threshold calibration. "same" = paraphrase (should hit),
# "diff" = similar wording but a different question (must NOT hit).
PAIRS = [
    ("same", "What does the Navigator LLM do in CoLLM-NAS?",
             "Explain the role of the Navigator LLM in CoLLM-NAS."),
    ("same", "What is DiffusionNAG and how does it generate architectures?",
             "How does DiffusionNAG generate neural architectures?"),
    ("same", "What design-space idea does RegNet introduce?",
             "Which design space concept is introduced by RegNet?"),
    ("same", "What is RZ-NAS and what problem does it address?",
             "What problem does RZ-NAS solve?"),
    ("same", "What are the key components of the CoLLM-NAS framework?",
             "Which main components make up CoLLM-NAS?"),
    ("diff", "What does the Navigator LLM do in CoLLM-NAS?",
             "What does the Generator LLM do in CoLLM-NAS?"),
    ("diff", "What is RZ-NAS and what problem does it address?",
             "What is DiffusionNAG and what problem does it address?"),
    ("diff", "What design-space idea does RegNet introduce?",
             "What design-space idea does RZ-NAS introduce?"),
    ("diff", "What are the key components of the CoLLM-NAS framework?",
             "What are the limitations of the CoLLM-NAS framework?"),
    ("diff", "How does DiffusionNAG generate architectures?",
             "How does DiffusionNAG evaluate architectures?"),
]


def calibrate():
    from retrieve import load_index_data
    model = load_index_data()[3]
    rows = []
    for kind, a, b in PAIRS:
        sim = float(_embed(model, a) @ _embed(model, b))
        rows.append((kind, sim))
        print(f"{kind:<5}{sim:.3f}  {a[:44]!r} vs {b[:44]!r}")
    same = [s for k, s in rows if k == "same"]
    diff = [s for k, s in rows if k == "diff"]
    print(f"\npara min={min(same):.3f}  diff max={max(diff):.3f}  "
          f"current threshold={THRESHOLD}")
    if min(same) > max(diff):
        print(f"separable: a threshold of about {(min(same) + max(diff)) / 2:.3f} works")
    else:
        print("NOT separable: some different questions score above some paraphrases; "
              "use a strict threshold (above diff max) and accept missing some paraphrases")


def demo():
    os.environ["USE_CACHE"] = "1"
    import time as _t
    from langgraph.types import Command
    import agentic_rag as ag
    app = ag.build_agentic_graph()
    ag._get_index()
    qs = ["What does the Navigator LLM do in CoLLM-NAS?",
          "Explain the role of the Navigator LLM in CoLLM-NAS.",
          "What does the Generator LLM do in CoLLM-NAS?",
          "What does the Navigator LLM do in CoLLM-NAS?"]
    for i, q in enumerate(qs):
        cfg = {"configurable": {"thread_id": f"cache-demo-{i}"}}
        t0 = _t.perf_counter()
        r = app.invoke(ag._initial_state(q), config=cfg)
        if "__interrupt__" in r:
            r = app.invoke(Command(resume="approve"), config=cfg)
        print(f"\nQ{i + 1}: {q}\n   {_t.perf_counter() - t0:.2f}s  path={r['path']}\n"
              f"   answer={r['answer'][:90]!r}")
    import semantic_cache as sc   # the instance agentic_rag used (this file runs as __main__)
    print(f"\ncache stats: {dict(sc.CACHE.stats)}")


if __name__ == "__main__":
    {"calibrate": calibrate, "demo": demo}.get(
        (sys.argv[1:] or [""])[0], lambda: print(__doc__))()