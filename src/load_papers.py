from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


PAPERS_FOLDER = Path("data/papers")


def load_all_papers():
    documents = []

    pdf_files = list(PAPERS_FOLDER.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        documents.extend(pages)

    print(f"\nTotal pages loaded: {len(documents)}")

    return documents


if __name__ == "__main__":
    documents = load_all_papers()