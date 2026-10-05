from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


PAPERS_FOLDER = Path("data/papers")
CHROMA_FOLDER = "results/chroma_db"


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


def create_vector_database(chunks):
    print("Loading Hugging Face embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_FOLDER,
        collection_name="research_papers"
    )

    print("Vector database created successfully!")
    print(f"Stored {len(chunks)} chunks.")

    return vectorstore


if __name__ == "__main__":
    documents = load_all_papers()
    chunks = split_documents(documents)
    vectorstore = create_vector_database(chunks)