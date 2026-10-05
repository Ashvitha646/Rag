from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from chunk_papers import load_all_papers, split_documents


CHROMA_FOLDER = "results/chroma_db_bge"


def load_vector_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return Chroma(
        persist_directory=CHROMA_FOLDER,
        collection_name="research_papers_bge",
        embedding_function=embeddings
    )


def create_bm25_index(chunks):
    documents = [chunk.page_content for chunk in chunks]

    tokenized_documents = [
        document.lower().split()
        for document in documents
    ]

    return BM25Okapi(tokenized_documents)


def hybrid_search(question, chunks, vectorstore, bm25, k=10):

    # Dense retrieval
    dense_results = vectorstore.similarity_search(
        question,
        k=k
    )

    # BM25 retrieval
    tokenized_question = question.lower().split()

    bm25_scores = bm25.get_scores(tokenized_question)

    bm25_indexes = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    bm25_results = [chunks[i] for i in bm25_indexes]

    # Combine and remove duplicates
    combined_results = dense_results + bm25_results

    unique_results = []
    seen_chunks = set()

    for chunk in combined_results:
        chunk_id = (
            chunk.metadata.get("source", ""),
            chunk.metadata.get("page", ""),
            chunk.page_content
        )

        if chunk_id not in seen_chunks:
            seen_chunks.add(chunk_id)
            unique_results.append(chunk)

    return unique_results


def rerank_results(question, results, top_k=3):

    print("\nLoading reranker model...")

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    pairs = [
        (question, result.page_content)
        for result in results
    ]

    scores = reranker.predict(pairs)

    scored_results = list(zip(results, scores))

    scored_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return scored_results[:top_k]


if __name__ == "__main__":

    print("Loading research papers...")

    documents = load_all_papers()
    chunks = split_documents(documents)

    print("Loading BGE ChromaDB...")

    vectorstore = load_vector_database()

    print("Creating BM25 index...")

    bm25 = create_bm25_index(chunks)

    question = input("\nEnter your question: ")

    print("\nRetrieving candidate chunks...")

    hybrid_results = hybrid_search(
        question,
        chunks,
        vectorstore,
        bm25,
        k=10
    )

    print(f"Retrieved {len(hybrid_results)} candidate chunks.")

    reranked_results = rerank_results(
        question,
        hybrid_results,
        top_k=3
    )

    print("\n" + "=" * 70)
    print("RERANKED RESULTS")
    print("=" * 70)

    for i, (result, score) in enumerate(reranked_results, start=1):

        source = Path(
            result.metadata.get("source", "Unknown")
        ).stem

        page = result.metadata.get("page", 0) + 1

        print(f"\n{i}. {source} — Page {page}")
        print(f"Reranker Score: {score:.4f}")
        print(result.page_content[:300])
        print("-" * 70)