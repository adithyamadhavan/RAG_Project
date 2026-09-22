from openai import OpenAI
from typing import List
import os

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed", timeout=60.0)


def generate_chunk_context(chunk: str, document_title: str, max_tokens: int = 80) -> str:
    """
    Generates a short (1-2 sentence) context blurb situating this chunk
    within its source document, per Anthropic's Contextual Retrieval technique.
    """
    prompt = f"""Document: {document_title}

Chunk:
{chunk}

Write a single short sentence (max 30 words) that situates this chunk within the document,
so it can be understood without seeing the rest of the document. Be concise and factual.
Do not repeat the chunk text itself.

Context sentence:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# def add_context_to_chunks(chunks: List[str], sources: List[str]) -> List[str]:
#     """
#     Prepends a generated context sentence to each chunk before embedding.
#     This is the core of contextual retrieval: the embedded text differs
#     from the displayed/retrieved text.
#     """
#     contextualized = []
#     for i, (chunk, source) in enumerate(zip(chunks, sources)):
#         context = generate_chunk_context(chunk, source)
#         contextualized.append(f"{context}\n\n{chunk}")
#         if (i + 1) % 20 == 0:
#             print(f"  contextualized {i + 1}/{len(chunks)} chunks...")
#     return contextualized

import random

def build_contextual_subset(chunked: dict, chunks_per_doc: int = 30, seed: int = 42) -> tuple:
    """
    Samples a subset of chunks (evenly across papers) and contextualizes them.
    Returns (contextualized_chunks, original_chunks, sources).
    """
    random.seed(seed)
    sampled_chunks = []
    sampled_sources = []

    for doc_name, chunks in chunked.items():
        n = min(chunks_per_doc, len(chunks))
        sample = random.sample(chunks, n)
        sampled_chunks.extend(sample)
        sampled_sources.extend([doc_name] * n)

    print(f"Sampled {len(sampled_chunks)} chunks across {len(chunked)} papers")

    contextualized = []
    for i, (chunk, source) in enumerate(zip(sampled_chunks, sampled_sources)):
        context = generate_chunk_context(chunk, source)
        contextualized.append(f"{context}\n\n{chunk}")
        if (i + 1) % 25 == 0:
            print(f"  contextualized {i + 1}/{len(sampled_chunks)} chunks...")

    return contextualized, sampled_chunks, sampled_sources

if __name__ == "__main__":
    from ingest import ingest_pdf_files
    from chunkers_recursive import chunk_corpus
    from pathlib import Path

    corpus = ingest_pdf_files(Path("corpus"))
    chunked = chunk_corpus(corpus)

    # Test on just the first 5 chunks of one paper before committing to the full corpus
    first_doc = next(iter(chunked))
    sample_chunks = chunked[first_doc][:5]

    print(f"Testing contextual chunking on 5 chunks from {first_doc}\n")
    for chunk in sample_chunks:
        context = generate_chunk_context(chunk, first_doc)
        print(f"Original: {chunk[:150]}")
        print(f"Context:  {context}")
        print(f"{'-'*80}")