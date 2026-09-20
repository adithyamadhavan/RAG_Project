import json
import time
from hybrid_retriever import hybrid_search
from reranker import rerank
from openai import OpenAI

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed",timeout=60)  # timeout in seconds

from run_eval import TEST_QUESTIONS
from generate import build_prompt


def generate_hybrid_answer(query: str, k: int = 10, final_k : int = 5) -> dict:
    candidates = hybrid_search(query, k=k, final_k=k)
    reranked = rerank(query, candidates, top_n=final_k)

    # build_prompt expects "text" key; hybrid/rerank results use "chunk"
    context_chunks = [{"source": r["source"], "chunk": r["chunk"]} for r in reranked]
    prompt = build_prompt(query, context_chunks)
    retrieved_chunks = [r["chunk"] for r in reranked]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400,
    )
    answer = response.choices[0].message.content
    cited_sources = sorted(set(r["source"] for r in reranked))
    return {"query": query, "answer": answer, "retrieved_sources": cited_sources,"retrieved_chunks": retrieved_chunks}


def run_hybrid_eval(output_path: str = "eval_results_hybrid.json"):
    results = []
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {question}")
        result = generate_hybrid_answer(question)
        results.append(result)
        print(f"→ {result['answer'][:150]}...")
        time.sleep(1)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_path}")
    return results


if __name__ == "__main__":
    run_hybrid_eval()
