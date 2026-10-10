# src/rag_plain.py
# Baseline for the supervisor-vs-swarm comparison: retrieve, then one LLM call.
# No graph, routing or approval. Same prompt, chunk slice, top-k and settings.
import os
from openai import OpenAI

from bench_utils import counted_chat, benchmark
from prompts import build_prompt

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed", timeout=60.0)

_INDEX = None


def _get_index():
    global _INDEX
    if _INDEX is None:
        from retrieve import load_index_data
        _INDEX = load_index_data()
    return _INDEX


QUERY = "What does the Navigator LLM do in CoLLM-NAS?"


def answer(query=QUERY):
    from retrieve import search
    index, chunks, sources, model = _get_index()
    results = search(query, index, chunks, sources, model, 3)
    findings = "\n\n".join(f"[{r['source']}] {r['chunk'][:300]}" for r in results)
    if os.environ.get("SHOW_CHUNKS"):
        print("CHUNKS:\n" + findings)
    prompt = build_prompt(query, findings)
    resp = counted_chat(
        client,
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=300,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    if os.environ.get("BENCH"):
        _get_index()  # load before timing
        benchmark(answer)
    else:
        print(f"\nFINAL ANSWER: {answer()}")