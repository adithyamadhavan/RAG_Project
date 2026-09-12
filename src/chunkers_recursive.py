from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List, Dict


def recursive_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Splits text using LangChain's RecursiveCharacterTextSplitter —
    prioritized ladder of separators, falls back to finer splits only
    when a chunk is still too large.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def chunk_corpus(corpus: Dict[str, str], chunk_size: int = 512, overlap: int = 50) -> Dict[str, List[str]]:
    """Applies recursive chunking across every document in the corpus."""
    return {
        name: recursive_chunk(text, chunk_size, overlap)
        for name, text in corpus.items()
    }



if __name__ == "__main__":
    from ingest import ingest_pdf_files

    corpus = ingest_pdf_files(Path("corpus"))
    chunked = chunk_corpus(corpus)

    first_doc_name = next(iter(corpus))
    chunks = chunked[first_doc_name]

    print(f"Document: {first_doc_name}")
    print(f"Original length: {len(corpus[first_doc_name])} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Avg chunk length: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")

    print("\n--- First 3 chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[Chunk {i}] ({len(chunk)} chars)")
        print(chunk[:200])