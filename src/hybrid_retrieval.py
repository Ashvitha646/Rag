from pathlib import Path
from rank_bm25 import BM25Okapi


def create_bm25_index(documents):
    """
    Create a BM25 keyword-search index from document chunks.
    """

    tokenized_documents = [
        document.page_content.lower().split()
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    print("BM25 index created successfully!")

    return bm25


def hybrid_retrieve(
    question,
    vectorstore,
    bm25,
    documents,
    k=10
):
    """
    Combine dense vector retrieval and BM25 keyword retrieval.

    Returns a list of unique document chunks.
    """

    # Dense/vector retrieval
    dense_results = vectorstore.similarity_search(
        question,
        k=k
    )

    # BM25 keyword retrieval
    tokenized_question = question.lower().split()

    bm25_scores = bm25.get_scores(tokenized_question)

    ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_results = [
        documents[i]
        for i in ranked_indices
        if bm25_scores[i] > 0
    ]

    # Combine dense and BM25 results
    combined_results = dense_results + bm25_results

    # Remove duplicate chunks
    unique_results = []
    seen = set()

    for document in combined_results:
        source = document.metadata.get("source", "")
        page = document.metadata.get("page", 0)
        content = document.page_content[:100]

        unique_id = (source, page, content)

        if unique_id not in seen:
            seen.add(unique_id)
            unique_results.append(document)

    return unique_results


if __name__ == "__main__":
    from chunk_papers import load_all_papers, split_documents
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    print("Loading research papers...")

    documents = load_all_papers()
    chunks = split_documents(documents)

    print("Loading BGE vector database...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory="results/chroma_db_bge",
        collection_name="research_papers_bge",
        embedding_function=embeddings
    )

    print("Creating BM25 index...")

    bm25 = create_bm25_index(chunks)

    question = "What is the purpose of LoRA?"

    print("\nRetrieving candidate chunks...")

    results = hybrid_retrieve(
        question=question,
        vectorstore=vectorstore,
        bm25=bm25,
        documents=chunks,
        k=10
    )

    print(f"Retrieved {len(results)} candidate chunks.")

    for i, document in enumerate(results[:3], start=1):
        source = Path(
            document.metadata.get("source", "Unknown")
        ).stem

        page = document.metadata.get("page", 0) + 1

        print(f"\n{i}. {source} — Page {page}")
        print(document.page_content[:300])