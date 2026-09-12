# RAG_Project
## Chunking Strategy Comparison (CoLLM-NAS paper)

| Metric | Recursive | Semantic |
|---|---|---|
| Chunk count | 110 | 21 |
| Avg chunk length | 412 chars | 2125 chars |

**Observation:** Semantic chunking produces far larger, topically coherent
chunks — e.g. the entire "Related Work" section (4334 chars) stayed intact
as one chunk, vs. recursive chunking splitting it into ~8-9 fragments at
arbitrary 500-char boundaries.

**Tradeoff:** Semantic chunking favors context completeness over retrieval
precision — better for broad/analytical questions, worse for narrow
factoid questions where a smaller, precise chunk is preferable.

**Decision:** [state which you're taking forward into Week 1's RAG pipeline, and why —
this depends on what kind of questions you expect your NAS literature assistant to answer]

**Note:** `langchain-experimental` (used for SemanticChunker) is flagged
as being sunset by LangChain — worth tracking for a future replacement.