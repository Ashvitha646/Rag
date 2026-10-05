import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from chunk_papers import load_all_papers, split_documents
from hybrid_retrieval import create_bm25_index, hybrid_retrieve


load_dotenv()


BGE_CHROMA_FOLDER = "results/chroma_db_bge"


def load_vector_database():
    """
    Load the existing BGE Chroma vector database.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=BGE_CHROMA_FOLDER,
        collection_name="research_papers_bge",
        embedding_function=embeddings
    )

    return vectorstore


def rerank_documents(question, documents, top_k=3):
    """
    Rerank retrieved chunks using a CrossEncoder model.
    """

    print("\nLoading reranker model...")

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    pairs = [
        [question, document.page_content]
        for document in documents
    ]

    scores = reranker.predict(pairs)

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results[:top_k]


def format_context(reranked_results):
    """
    Prepare retrieved chunks for the LLM prompt.
    """

    context_parts = []

    for i, (document, score) in enumerate(
        reranked_results,
        start=1
    ):
        source = Path(
            document.metadata.get("source", "Unknown")
        ).stem

        page = document.metadata.get("page", 0) + 1

        context_parts.append(
            f"[Source {i}: {source}, Page {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


def format_sources(reranked_results):
    """
    Display the top three paper sources.
    """

    sources = []

    for i, (document, score) in enumerate(
        reranked_results,
        start=1
    ):
        source = Path(
            document.metadata.get("source", "Unknown")
        ).stem

        page = document.metadata.get("page", 0) + 1

        sources.append(
            f"{i}. {source} — Page {page}"
        )

    return "\n".join(sources)


def answer_question(question):

    print("\nLoading research papers...")
    documents = load_all_papers()

    print("Splitting documents into chunks...")
    chunks = split_documents(documents)

    print("Loading BGE vector database...")
    vectorstore = load_vector_database()

    print("Creating BM25 index...")
    bm25 = create_bm25_index(chunks)

    print("\nRetrieving candidate chunks...")

    candidate_documents = hybrid_retrieve(
        question=question,
        vectorstore=vectorstore,
        bm25=bm25,
        documents=chunks,
        k=10
    )

    print(
        f"Retrieved {len(candidate_documents)} candidate chunks."
    )

    reranked_results = rerank_documents(
        question=question,
        documents=candidate_documents,
        top_k=3
    )

    print("\n" + "=" * 70)
    print("RERANKED RESULTS")
    print("=" * 70)

    for i, (document, score) in enumerate(
        reranked_results,
        start=1
    ):
        source = Path(
            document.metadata.get("source", "Unknown")
        ).stem

        page = document.metadata.get("page", 0) + 1

        print(f"\n{i}. {source} — Page {page}")
        print(f"Reranker Score: {score:.4f}")
        print(document.page_content[:300])
        print("-" * 70)

    context = format_context(reranked_results)
    sources = format_sources(reranked_results)

    print("\nGenerating answer using Groq...")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    prompt = f"""
You are a research paper question-answering assistant.

Answer the user's question using only the provided research paper context.

Rules:
1. Give a clear and simple explanation.
2. Do not invent information.
3. If the answer is not available in the context, say:
   "I could not find enough information in the indexed research papers."
4. Refer to sources using [Source 1], [Source 2], or [Source 3] only when supported by the context.
5. Do not create fake citations or references.
6. Keep the answer concise but informative.

User Question:
{question}

Research Paper Context:
{context}
"""

    response = llm.invoke(prompt)

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(response.content)

    print("\n" + "=" * 70)
    print("TOP 3 SOURCES")
    print("=" * 70)

    print(sources)
    return response.content, reranked_results


if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    ).strip()

    if question:
        answer_question(question)
    else:
        print("Please enter a question.")