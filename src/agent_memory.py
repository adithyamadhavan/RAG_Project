from pathlib import Path
from datetime import datetime
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
INDEX_PATH = Path("agent_memory_index.bin")
DATA_PATH = Path("agent_memory_data.pkl")


class LongTermMemory:
    """Persists agent interactions across process restarts, retrievable by
    semantic similarity. Same mechanism as the RAG pipeline (FAISS +
    embeddings), applied to the agent's own history instead of a corpus."""

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.memories = []
        if INDEX_PATH.exists() and DATA_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(DATA_PATH, "rb") as f:
                self.memories = pickle.load(f)
            print(f"[memory] loaded {len(self.memories)} existing memories")
        else:
            dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dim)
            print("[memory] starting empty")

    def add(self, text: str, metadata: dict = None):
        vec = self.model.encode([text], convert_to_numpy=True).astype("float32")
        self.index.add(vec)
        self.memories.append({
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        })
        faiss.write_index(self.index, str(INDEX_PATH))
        with open(DATA_PATH, "wb") as f:
            pickle.dump(self.memories, f)

    def retrieve(self, query: str, k: int = 3) -> list:
        if not self.memories:
            return []
        k = min(k, len(self.memories))
        vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        _, indices = self.index.search(vec, k)
        return [self.memories[i] for i in indices[0]]

    def retrieve_with_scores(self, query: str, k: int = 3) -> list:
        """Same as retrieve(), but also returns the FAISS distance for each
        result, so we can see how 'close' a match actually is before deciding
        whether to trust it."""
        if not self.memories:
            return []
        k = min(k, len(self.memories))
        vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(vec, k)
        return [
            {"memory": self.memories[i], "distance": float(d)}
            for d, i in zip(distances[0], indices[0])
        ]


if __name__ == "__main__":
    mem = LongTermMemory()

    mem.add(
        "User asked what 15 * 23 + 7 equals. The calculator tool returned 352.",
        metadata={"topic": "calculation"},
    )
    mem.add(
        "User asked about the Navigator LLM in CoLLM-NAS. Retrieved corpus text "
        "describes it as providing search strategy via comprehension of the search "
        "space and exploration strategy.",
        metadata={"topic": "collm_nas"},
    )

    print(f"\ntotal memories now: {len(mem.memories)}")

    print("\n--- retrieving for a related new query ---")
    for r in mem.retrieve("what did we learn about CoLLM-NAS before?", k=1):
        print(f"[{r['timestamp']}] {r['text']}")