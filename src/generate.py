from openai import OpenAI
from retrieve import load_index_data,search


VLLM_base_url = "http://127.0.0.1:8000/v1"
Model_name = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

client = OpenAI(base_url=VLLM_base_url,api_key="not-needed",timeout=60)  # timeout in seconds


def build_prompt(query: str, retrieved_chunks: list) -> str:
    """
    Formats retrieved chunks + the user's question into a prompt that
    instructs the model to answer ONLY from the provided context and
    cite sources explicitly.
    """
    context_blocks = []
    for r in retrieved_chunks:
        context_blocks.append(f"[Source: {r['source']}]\n{r['chunk']}")
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a research assistant answering questions about neural architecture search (NAS) papers.

Answer the question using ONLY the context provided below. If the context doesn't contain enough information to answer, say so explicitly — do not use outside knowledge.

When you use information from a source, cite it by filename in brackets, e.g. [RZ_NAS.pdf].

Context:
{context}

Question: {query}

Answer:"""
    return prompt

def generate_answer(query: str, index, chunks, sources, model, k: int = 5) -> dict:
    retrieved = search(query, index, chunks, sources, model, top_k=k)
    prompt = build_prompt(query, retrieved)
    response = client.chat.completions.create(
        model=Model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=400,
    )
    answer = response.choices[0].message.content
    cited_sources = sorted(set(r["source"] for r in retrieved))
    return {"query": query, "answer": answer, "retrieved_sources": cited_sources}


import time

def generate_with_retry(query, *args, max_retries=2, **kwargs):
    for attempt in range(max_retries + 1):
        try:
            return generate_answer(query, *args, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                print(f"  Retry {attempt+1} after error: {e}")
                time.sleep(5)
            else:
                raise

if __name__ == "__main__":
    test_questions = [
        "What is quality diversity in neural architecture search?",
        "How does CoLLM-NAS coordinate between multiple LLMs?",
        "What is RegNet's design space?",
    ]

    for q in test_questions:
        result = generate_answer(q)
        print(f"\n{'='*80}")
        print(f"Q: {result['query']}")
        print(f"{'='*80}")
        print(f"A: {result['answer']}")
        print(f"\nSources used: {result['retrieved_sources']}")
