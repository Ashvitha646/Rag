from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from chunk_papers import load_all_papers, split_documents


BGE_CHROMA_FOLDER = "results/chroma_db_bge"


def create_bge_vector_database():
    print("Loading research papers...")

    documents = load_all_papers()
    chunks = split_documents(documents)

    print("Loading BGE embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Creating BGE ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=BGE_CHROMA_FOLDER,
        collection_name="research_papers_bge"
    )

    print("BGE ChromaDB created successfully!")
    print(f"Stored {len(chunks)} chunks.")

    return vectorstore


if __name__ == "__main__":
    create_bge_vector_database()