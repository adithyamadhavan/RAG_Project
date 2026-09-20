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

## Week 2: Quantified retrieval comparison (naive vs. hybrid+rerank)

After Week 1's eyeballed comparisons proved inconclusive, all 20 test questions were
hand-labeled for chunk-level relevance (`relevance_labels.json`) against both
pipelines' actual retrieved chunks, and scored with MRR and NDCG@5
(`compute_final_metrics.py`).

| Metric | Naive (dense-only) | Hybrid + Rerank | Change |
|---|---|---|---|
| MRR | 0.3458 | 0.3292 | −4.8% |
| NDCG@5 | 0.5092 | 0.4835 | −5.0% |

**Finding: hybrid search + reranking performed measurably worse than naive dense
retrieval on this corpus and question set — a real, unexpected result, not noise
(both independent metrics agree in direction and magnitude).**

Per-question analysis (`analyze_regression.py`) shows the effect is broad and mild,
not one catastrophic failure:
- **6 questions regressed** — in every case the relevant chunk's rank shifted by
  exactly +1 (e.g. rank 3 → 4)
- **3 questions improved** — in every case by exactly −1 (e.g. rank 3 → 2)
- **11 questions unchanged**
- **1 question (a deliberately speculative, cross-domain synthesis question) lost
  its relevant chunk entirely** from hybrid+rerank's top-5 — notably, the one
  complete miss occurred on the hardest, most ambiguous question in the set

**Interpretation:** dense embedding similarity alone was already reasonably
well-calibrated for this corpus of 10 dense academic papers. Reranking
(`bge-reranker-base`) appears to weakly favor longer, more technical-sounding
passages over concise, directly-answering ones — consistent with a pattern
observed during manual relevance labeling, where citation-list entries or
tangentially related passages sometimes scored above the chunk that actually
answered the question.

**This result should not be generalized beyond this specific setup** — 20 questions
over 10 papers is a small test bed, and the finding is corpus- and
question-distribution-specific. It does, however, demonstrate the value of
measuring rather than assuming: added retrieval complexity (BM25 fusion, RRF,
cross-encoder reranking) is not automatically an improvement, and the only way to
know is to score it.

**Follow-up directions this motivates** (not yet implemented):
- Test an alternative or larger reranker model (e.g. `bge-reranker-large`) for
  better calibration
- Narrow the candidate shortlist fed into reranking (currently k=10) to reduce
  exposure to tangentially-relevant "noise" chunks
- Revisit whether RRF fusion parameters (k=60) are well-tuned for this corpus size

## RAGAS: attempted, reverted due to dependency conflicts

RAGAS was evaluated as a second, LLM-as-judge-based eval framework (faithfulness,
answer relevancy, context precision/recall) to complement the MRR/NDCG scoring
above. Installation repeatedly caused severe dependency conflicts with the
project's core LangChain-based pipeline — `ragas` requires `langchain-core`
versions incompatible with `langgraph`, `langchain-huggingface`, and
`langchain-experimental`, all used elsewhere in this project. Two install attempts
(latest RAGAS, then a pinned older version) both broke the main pipeline's imports,
requiring a full dependency restoration each time.

**Decision: RAGAS was dropped in favor of the custom MRR/NDCG implementation
above.** This was a deliberate tradeoff to avoid ongoing environment instability
close to a project deadline, not an oversight — a legitimate real-world engineering
call when a tool's dependency footprint threatens a working system for marginal
additional signal. A production system wanting both RAGAS-style faithfulness
scoring and a stable core pipeline would isolate RAGAS in its own environment
(e.g. a separate venv or container), calling it only against already-saved eval
JSON files rather than integrating it into the main dependency tree.

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