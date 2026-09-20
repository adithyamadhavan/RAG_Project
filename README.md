# RAG Project — Week 1: Naive RAG → Hybrid Search & Reranking

A retrieval-augmented generation pipeline built over a 10-paper Neural Architecture
Search (NAS) literature corpus, served entirely locally via `vllm-mlx` (Apple Silicon
MLX backend for vLLM) running Qwen2.5-1.5B-Instruct.

## Architecture

```
PDF corpus (10 NAS papers)
  → ingest.py        (PyMuPDF extraction + text cleaning)
  → chunkers_latest.py (recursive chunking, LangChain) / semantic_chunker.py
  → embed_index.py   (bge-small-en-v1.5 embeddings → FAISS flat L2 index)
  → bm25_retriever.py (BM25 sparse index, rank-bm25)
  → hybrid_retriever.py (Reciprocal Rank Fusion, k=60)
  → reranker.py      (bge-reranker-base cross-encoder)
  → generate.py      (prompt construction → vllm-mlx → grounded answer w/ citations)
```

## Chunking: Recursive vs Semantic

| Metric | Recursive | Semantic |
|---|---|---|
| Chunk count (single paper) | 110 | 21 |
| Avg chunk length | 412 chars | 2125 chars |

Semantic chunking (LangChain's `SemanticChunker`) produces far larger, topically
coherent chunks — e.g. an entire "Related Work" section stayed intact as one 4334-char
chunk, vs. recursive chunking splitting it into ~8-9 fragments at arbitrary boundaries.

**Decision: recursive chunking taken forward as the default.** Semantic chunking
favors context completeness over retrieval precision; recursive's smaller, more
numerous chunks give finer-grained retrieval control, which matters more for a corpus
expected to answer both narrow factoid and broader analytical questions. Semantic
chunking's dependency (`langchain-experimental`) is also flagged as being sunset by
LangChain, a secondary factor against committing to it long-term.

## Data quality issue: prompt-injection artifacts in source PDFs

PyMuPDF extraction of `RZ_NAS.pdf` surfaced literal chat-template tokens
(`<|im_start|>user ... <|im_end|>`) embedded in the paper's text — likely from the
paper's own methodology section showing example LLM prompts. These artifacts were
retrieved as legitimate content by both dense and BM25 search (appearing in each
method's individual top-3 for a test query), and by the reranker.

**Fix:** added regex-based sanitization at ingestion time (`ingest.py`), stripping
`<|im_start|>...<|im_end|>` spans and stray tokens. Verified clean across the full
pipeline after rebuilding both FAISS and BM25 indices — corpus chunk count dropped
1258 → 1255, confirming removal. This is exactly the class of problem the guardrails
work planned for Week 2 is meant to generalize and catch systematically, rather than
patching per-artifact as they're discovered.

## Retrieval quality: known weak case (RegNet "design space")

Across all three retrieval configurations tested (naive dense, BM25, hybrid+RRF), a
query about "RegNet's design space" consistently surfaced architectural/parameter
details rather than content that actually explains "design space" as a concept — the
corpus's RegNet paper may not contain a clearly extractable section on this specific
term. Hybrid+RRF fusion partially improved this (surfaced more conceptually relevant
content than either method alone at rank 4), but did not fully resolve it.

The naive pipeline handled this gap by explicitly stating insufficient context; the
hybrid+rerank pipeline's answer sounded more confident and specific but was not
independently verified against the source text for accuracy — flagged here as an open
question rather than a confirmed hallucination, and a good candidate for closer
inspection in Week 2's eval work.

## Grounding behavior: non-deterministic

Identical questions posed across different runs sometimes produced different
groundedness behavior — the model occasionally answered confidently from
tangentially-related context on one run, and correctly stated insufficient
information on another, for the same underlying corpus. This is a known RAG failure
mode: the model conflates "I have related context" with "I have the answer."
Motivates Week 2's corrective RAG (grade-and-re-query loop) and structured-output
guardrails work.

## Infrastructure: vllm-mlx runaway generation

A broad, multi-paper synthesis question (Q16 of the 20-question test set) reliably
triggered unbounded/repetitive generation on the 1.5B model, causing indefinite hangs
(600+ seconds) on `vllm-mlx`. Confirmed via repeated testing that this was
query-specific and reproducible even after full server restarts — ruling out a
server-side resource leak as the cause.

**Fix:** added a `max_tokens=400` generation cap and a 60-second client-side timeout.
Resolved the issue completely — all 20 questions now run cleanly end-to-end on both
pipelines. Documented here as a known limitation of the community MLX backend
(`vllm-mlx`) rather than an issue with vLLM's CUDA implementation; a production
deployment on CUDA hardware would likely not exhibit this specific failure mode, but
the `max_tokens` cap is good practice regardless.

## Eval harness

- `tests/test_questions.md` — 20 hand-written questions spanning factoid, analytical,
  and deliberately hard/ambiguous tiers (the latter designed to test grounding and
  "insufficient information" honesty)
- `eval_results_naive.json` / `eval_results_hybrid.json` — full 20-question outputs
  for the naive and hybrid+rerank pipelines respectively, both run on the sanitized
  corpus post-fix
- `eval_metrics.py` — MRR and NDCG@5 implementations, verified against a toy example
  (MRR: 0.5000, NDCG@5: 0.5414). **Not yet applied to the full 20-question set** —
  relevance labeling was scoped out of Week 1 as a time tradeoff; eyeballed
  comparisons between the two result files show mixed outcomes (e.g. a clear
  improvement on the RegNet query, a possible regression on the quality-diversity
  query), which is exactly why quantified scoring is flagged as a priority for
  Week 2 rather than relying on spot-checks going forward.

## Known technical debt

- Inconsistent field naming (`chunk` vs `text`) across `retrieve.py`, `bm25_retriever.py`,
  `hybrid_retriever.py`, and `generate.py` caused repeated integration bugs during
  development. Worth a standardization pass before Week 3 adds further modules.
- `langchain-community` (used for an unused FAISS import) and `langchain-experimental`
  (semantic chunker) are both flagged as sunset/deprecated by LangChain — worth
  tracking for replacement.
- Hand-rolled recursive chunker (`chunkers_manual.py`) retained for reference — three
  distinct bugs debugged during development (separator-fallback logic, default
  argument masking the fallback ladder, resulting oversized chunks) before switching
  to LangChain's production implementation.

## Stack

- **Inference:** vllm-mlx (Qwen2.5-1.5B-Instruct-4bit), OpenAI-compatible local endpoint
- **Embeddings:** BAAI/bge-small-en-v1.5 (sentence-transformers)
- **Vector store:** FAISS (IndexFlatL2)
- **Sparse retrieval:** BM25 (rank-bm25)
- **Fusion:** Reciprocal Rank Fusion, k=60
- **Reranking:** BAAI/bge-reranker-base (cross-encoder)
- **PDF extraction:** PyMuPDF
- **Chunking:** LangChain RecursiveCharacterTextSplitter (production), SemanticChunker (evaluated)