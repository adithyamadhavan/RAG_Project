import fitz 
from pathlib import Path
from typing import List, Optional , Dict

def extract_text_from_pdf(pdf_path: str) -> List[str]:
    """
    Extracts text from a PDF file.
    Returns text from each page of the PDF as a list of strings.
    Args:
        pdf_path (str): The path to the PDF file.
    """
    text_list =[]
    docs = fitz.open(pdf_path)
    for pages in docs:
        text = pages.get_text()
        text_list.append(text)
    docs.close()
    return "\n".join(text_list)


def ingest_pdf_files(pdf_dir: Path) -> Dict[str, str]:
    """
    Ingests all PDF files in a directory and extracts text from them.
    Returns a dictionary with file names as keys and extracted text as values.
    Args:
        pdf_dir (Path): The path to the directory containing PDF files.
    """
    results = {}
    pdf_path = list(pdf_dir.glob("*.pdf"))
    if not pdf_path:
        raise FileNotFoundError(f"No PDF files found in directory: {pdf_dir}")
    for pdf_file in pdf_path:
        try:
            text = extract_text_from_pdf(str(pdf_file))
            results[pdf_file.name] = text
            print(f"Successfully extracted text from {pdf_file.name}")
        except Exception as e:
            print(f"Error extracting text from {pdf_file.name}: {e}")
    return results

if __name__ == "__main__":
    pdf_dir = Path("corpus")
    corpus = ingest_pdf_files(pdf_dir)
    print(f"Extracted text from {len(corpus)} PDF files.")
    if corpus:
        first_doc = next(iter(corpus))
        print(f"First document: {first_doc}")
        print(corpus[first_doc][:500])  # Print first 500 characters of the first document
        print(f"Total characters in first document: {len(corpus[first_doc])}")
        