from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


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

    print(f"Total pages loaded: {len(documents)}")

    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


def create_embeddings():
    print("Loading Hugging Face embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded successfully.")

    return embeddings


if __name__ == "__main__":
    documents = load_all_papers()
    chunks = split_documents(documents)
    embeddings = create_embeddings()

    sample_text = chunks[0].page_content

    vector = embeddings.embed_query(sample_text)

    print("\nEmbedding created successfully!")
    print(f"Embedding dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")