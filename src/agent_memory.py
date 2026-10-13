# src/agent_memory.py
# Long-term memory: past interactions stored as embeddings (FAISS) and
# retrieved by semantic similarity. Persisted to disk across restarts.
#
# Changes from the first version (the public methods are unchanged, so
# langgraph_agent.py keeps working):
#   - dedupe-on-write: a new memory within DUP_DISTANCE of an existing one
#     (same user) UPDATES that entry instead of adding a second copy
#   - user_id scoping: memories belong to a user and are only retrieved for them
#   - clear() / delete support, and a model can be passed in to avoid loading
#     a second copy of the embedding model
#   - add() now returns "added" or "updated" so callers/tests can see what happened
#
# Distances are FAISS L2 on bge-small embeddings (unit length, so range 0..2;
# 0 = identical text). Defaults: DUP_DISTANCE 0.15, relevance 0.5 (as before).
import os
import pickle
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MEMORY_DIR = Path(os.environ.get("MEMORY_DIR", "."))
INDEX_PATH = MEMORY_DIR / "agent_memory_index.bin"
DATA_PATH = MEMORY_DIR / "agent_memory_data.pkl"
DUP_DISTANCE = float(os.environ.get("MEMORY_DUP_DISTANCE", "0.15"))
RELEVANCE_DISTANCE = float(os.environ.get("MEMORY_RELEVANCE", "0.5"))


class LongTermMemory:
    def __init__(self, model=None, index_path=None, data_path=None):
        self.model = model or SentenceTransformer(MODEL_NAME)
        self.index_path = Path(index_path or INDEX_PATH)
        self.data_path = Path(data_path or DATA_PATH)
        self.memories = []
        if self.index_path.exists() and self.data_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.data_path, "rb") as f:
                self.memories = pickle.load(f)
            for m in self.memories:               # older files had no user_id / count
                m.setdefault("user_id", "default")
                m.setdefault("count", 1)
            print(f"[memory] loaded {len(self.memories)} existing memories")
        else:
            self.index = faiss.IndexFlatL2(self._dim())
            print("[memory] starting empty")

    # ------------------------------------------------------------ internals
    def _dim(self):
        fn = getattr(self.model, "get_embedding_dimension", None) \
            or self.model.get_sentence_embedding_dimension
        return fn()

    def _embed(self, text):
        return self.model.encode([text], convert_to_numpy=True).astype("float32")

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.data_path, "wb") as f:
            pickle.dump(self.memories, f)

    def _search(self, vec, k, user_id=None):
        """Nearest memories for this user: list of (position, distance)."""
        if not self.memories:
            return []
        n = len(self.memories)
        distances, indices = self.index.search(vec, n if user_id else min(k, n))
        out = []
        for d, i in zip(distances[0], indices[0]):
            if i < 0:
                continue
            if user_id and self.memories[i].get("user_id") != user_id:
                continue
            out.append((int(i), float(d)))
            if len(out) == k:
                break
        return out

    def _replace_vector(self, pos, vec):
        """IndexFlatL2 cannot overwrite a row, so rebuild it with row `pos` swapped."""
        vecs = self.index.reconstruct_n(0, self.index.ntotal)
        vecs[pos] = vec[0]
        self.index = faiss.IndexFlatL2(vecs.shape[1])
        self.index.add(vecs)

    # ------------------------------------------------------------ public API
    def add(self, text: str, metadata: dict = None, user_id: str = "default",
            key: str = None) -> str:
        """`key` is the text that gets embedded and compared (for Q&A memories:
        the question). Default is the whole text, as before."""
        vec = self._embed(key or text)
        nearest = self._search(vec, 1, user_id)
        if nearest and nearest[0][1] < DUP_DISTANCE:        # dedupe-on-write
            pos = nearest[0][0]
            m = self.memories[pos]
            m.update(text=text, timestamp=datetime.now().isoformat(),
                     metadata=metadata or m["metadata"], count=m["count"] + 1)
            self._replace_vector(pos, vec)
            self._save()
            return "updated"
        self.index.add(vec)
        self.memories.append({
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "user_id": user_id,
            "count": 1,
        })
        self._save()
        return "added"

    def retrieve(self, query: str, k: int = 3, user_id: str = None) -> list:
        return [r["memory"] for r in self.retrieve_with_scores(query, k, user_id)]

    def retrieve_with_scores(self, query: str, k: int = 3, user_id: str = None) -> list:
        """Memories plus their FAISS distance, so callers can apply a relevance
        threshold instead of trusting a nearest neighbour that is not close."""
        return [{"memory": self.memories[i], "distance": d}
                for i, d in self._search(self._embed(query), k, user_id)]

    def clear(self, user_id: str = None):
        """Forget everything, or only one user's memories."""
        keep = [m for m in self.memories if user_id and m.get("user_id") != user_id]
        vecs = None
        if keep and self.index.ntotal:
            all_vecs = self.index.reconstruct_n(0, self.index.ntotal)
            vecs = np.stack([all_vecs[i] for i, m in enumerate(self.memories)
                             if user_id and m.get("user_id") != user_id])
        self.index = faiss.IndexFlatL2(self._dim())
        if vecs is not None:
            self.index.add(vecs)
        self.memories = keep
        self._save()


# ---------------------------------------------------------------- helpers for graphs
# Two functions any graph can call. Memory is created lazily, once per process.
_LTM = None


def get_memory(model=None) -> "LongTermMemory":
    global _LTM
    if _LTM is None:
        _LTM = LongTermMemory(model=model)
    return _LTM


def recall(question: str, model=None, user_id: str = "default", k: int = 2) -> list:
    """Texts of past interactions close enough to be worth showing."""
    hits = get_memory(model).retrieve_with_scores(question, k=k, user_id=user_id)
    return [h["memory"]["text"] for h in hits if h["distance"] < RELEVANCE_DISTANCE]


def store_turn(question: str, answer: str, model=None, user_id: str = "default") -> str:
    text = f"User asked: {question} Answer given: {answer[:300]}"
    # Embed only the question, so the same question with a slightly different
    # answer is still recognised as a duplicate, and recall matches question to question.
    return get_memory(model).add(text, metadata={"kind": "qa"}, user_id=user_id,
                                 key=f"User asked: {question}")


if __name__ == "__main__":
    mem = LongTermMemory()

    for text, topic in [
        ("User asked what 15 * 23 + 7 equals. The calculator tool returned 352.", "calculation"),
        ("User asked about the Navigator LLM in CoLLM-NAS. Retrieved corpus text "
         "describes it as providing search strategy via comprehension of the search "
         "space and exploration strategy.", "collm_nas"),
    ]:
        print(mem.add(text, metadata={"topic": topic}), "-", text[:50])
    # same thing again: should be "updated", not a new row
    print(mem.add("User asked what 15 * 23 + 7 equals. The calculator tool returned 352.",
                  metadata={"topic": "calculation"}))

    print(f"\ntotal memories now: {len(mem.memories)}")
    print("\n--- retrieving for a related new query ---")
    for r in mem.retrieve_with_scores("what did we learn about CoLLM-NAS before?", k=1):
        print(f"[{r['memory']['timestamp']}] d={r['distance']:.3f} {r['memory']['text']}")