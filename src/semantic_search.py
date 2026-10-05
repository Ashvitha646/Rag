from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


CHROMA_FOLDER = "results/chroma_db"


def load_vector_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_FOLDER,
        embedding_function=embeddings,
        collection_name="research_papers"
    )

    return vectorstore


def search_papers(query, top_k=3):
    vectorstore = load_vector_database()

    results = vectorstore.similarity_search_with_score(
        query,
        k=top_k
    )

    print(f"\nQuestion: {query}")
    print(f"Top {top_k} relevant results:\n")

    for rank, (document, score) in enumerate(results, start=1):
        print("=" * 70)
        print(f"Result {rank}")
        print(f"Similarity score: {score:.4f}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Page number: {document.metadata.get('page', 0) + 1}")
        print("\nContent:")
        print(document.page_content[:700])


if __name__ == "__main__":
    question = input("Enter your research question: ")
    search_papers(question, top_k=3)