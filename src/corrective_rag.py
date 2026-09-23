from openai import OpenAI
import os
from typing import List, Dict
from generate import build_prompt
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed", timeout=60.0)





# def grade_context_sufficiency(query: str, retrieved_chunks: List[str]) -> bool:
#     """
#     Asks the LLM to judge whether retrieved context is sufficient to answer
#     the query, BEFORE generation happens. Returns True/False.
#     """
#     context = "\n\n---\n\n".join(retrieved_chunks)

#     prompt = f"""Question: {query}

# Retrieved context:
# {context}

# Does the retrieved context above contain enough specific information to directly
# and confidently answer the question? Answer with exactly one word: YES or NO.

# Answer:"""

#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0,
#         max_tokens=5,
#     )
#     answer = response.choices[0].message.content.strip().upper()
#     return answer.startswith("YES")

def grade_context_sufficiency(query: str, retrieved_chunks: List[str]) -> bool:
    context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""Question: {query}

Retrieved context:
{context}

Step 1: List the specific facts in the context above that are DIRECTLY relevant to answering the question. If there are none, write "No directly relevant facts found."

Step 2: Based ONLY on Step 1, can the question be answered with specific, confident detail (not a vague or general statement)? Answer with exactly one word on a new line: YES or NO.

Step 1:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,  # needs more room now, since it reasons first
    )
    raw_answer = response.choices[0].message.content
    print(f"    [DEBUG] raw model output:\n{raw_answer}\n")

    # Look for YES/NO specifically, not just first word — the reasoning comes first now
    answer_upper = raw_answer.strip().upper()
    if "NO" in answer_upper.split("\n")[-1] or answer_upper.rstrip().endswith("NO"):
        return False
    return "YES" in answer_upper.split("\n")[-1] or answer_upper.rstrip().endswith("YES")

from generate import build_prompt, client#, MODEL_NAME


def generate_with_crag(query: str, index, chunks, sources, model, k: int = 5) -> dict:
    """
    Full corrective RAG flow: retrieve, grade sufficiency, then either
    generate normally or return an explicit insufficient-information response.
    """
    from retrieve import search
    retrieved = search(query, index, chunks, sources, model, top_k=k)
    retrieved_text = [r["chunk"] for r in retrieved]

    sufficient = grade_context_sufficiency(query, retrieved_text)

    if not sufficient:
        return {
            "query": query,
            "answer": "I don't have enough specific information in the available sources to answer this question confidently.",
            "retrieved_sources": sorted(set(r["source"] for r in retrieved)),
            "crag_triggered": True,
        }

    # Context judged sufficient — proceed to normal generation
    prompt = build_prompt(query, retrieved)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400,
    )
    answer = response.choices[0].message.content
    return {
        "query": query,
        "answer": answer,
        "retrieved_sources": sorted(set(r["source"] for r in retrieved)),
        "crag_triggered": False,
    }




if __name__ == "__main__":
    from retrieve import load_index_data

    index, chunks, sources, model = load_index_data()

    test_cases = [
        "How does CoLLM-NAS coordinate between multiple LLMs?",
        "What is RegNet's design space?",
    ]

    for query in test_cases:
        result = generate_with_crag(query, index, chunks, sources, model)
        print(f"\nQuery: {query}")
        print(f"CRAG triggered (insufficient context): {result['crag_triggered']}")
        print(f"Answer: {result['answer'][:300]}")