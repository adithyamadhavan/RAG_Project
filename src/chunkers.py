from pathlib import Path
from typing import List, Dict


def recursive_chunk(
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        separator: List[str] = ["\n"],
) -> List[str]:
    """
    Splits texts into chunks of a specified size with optional overlap and separator.
    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk. Default is 1000 characters.
        overlap (int): The number of overlapping characters between chunks. Default is 200 characters.
        separator (List[str]): A list of separators to use for splitting the text. Default is ["\n"].
    """
    if separator is None:
        separator = ["\n\n","\n"," "," "]

    def split_text(text: str, separator: List[str]) -> List[str]:
        if not separator:
            return [text]
        sep = separator[0]
        parts = text.split(sep)
        chunks = []
        current_chunk = ""
        for part in parts:
            candidate = current_chunk + (sep if current_chunk else "") + part
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(part) > chunk_size:
                        chunks.extend(split_text(part, separator[1:]))
                        current_chunk = ""
                else:
                        current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    raw_text_chunks = split_text(text, separator)
    overlapped = []
    for i, chunk in enumerate(raw_text_chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prev_tail = raw_text_chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + chunk)

    return overlapped

if __name__ == "__main__":
    from ingest import ingest_pdf_files

    corpus = ingest_pdf_files(Path("corpus"))
    first_doc_name = next(iter(corpus))
    text = corpus[first_doc_name]

    chunks = recursive_chunk(text, chunk_size=500, overlap=75)

    print(f"Document: {first_doc_name}")
    print(f"Total chars: {len(text)}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Avg chunk length: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")

    print("\n--- First 3 chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n[Chunk {i}] ({len(chunk)} chars)")
        print(chunk[:200])

