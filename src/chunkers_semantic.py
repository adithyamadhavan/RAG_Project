from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from typing import List, Dict


def semantic_chunk(text: str, threshold_type: str = "percentile") -> List[str]:
    """
    Splits text based on semantic similarity between sentences,
    rather than a fixed character count. Splits occur where
    similarity between consecutive sentences drops sharply.
    """
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    splitter = SemanticChunker(embeddings, breakpoint_threshold_type=threshold_type)
    return splitter.split_text(text)


if __name__ == "__main__":
    from ingest import ingest_pdf_files

    corpus = ingest_pdf_files(Path("corpus"))
    first_doc_name = next(iter(corpus))
    text = corpus[first_doc_name]

    chunks = semantic_chunk(text)

    print(f"Document: {first_doc_name}")
    print(f"Original length: {len(text)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Avg chunk length: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")

    print("\n--- First 3 chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[Chunk {i}] ({len(chunk)} chars)")
        print(chunk[:200])
        