from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


MINILM_CHROMA_FOLDER = "results/chroma_db"
BGE_CHROMA_FOLDER = "results/chroma_db_bge"


def load_minilm_database():
    print("Loading MiniLM database...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=MINILM_CHROMA_FOLDER,
        collection_name="research_papers",
        embedding_function=embeddings
    )

    return vectorstore


def load_bge_database():
    print("Loading BGE database...")

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


def display_results(model_name, results):
    print("\n" + "=" * 70)
    print(f"{model_name} RESULTS")
    print("=" * 70)

    for i, (document, score) in enumerate(results, start=1):
        source = Path(
            document.metadata.get("source", "Unknown")
        ).stem

        page = document.metadata.get("page", 0) + 1

        print(f"\n{i}. {source} — Page {page}")
        print(f"Distance Score: {score:.4f}")
        print(document.page_content[:300])
        print("-" * 70)


def compare_models(question):
    minilm_db = load_minilm_database()
    bge_db = load_bge_database()

    minilm_results = minilm_db.similarity_search_with_score(
        question,
        k=3
    )

    bge_results = bge_db.similarity_search_with_score(
        question,
        k=3
    )

    print("\n" + "#" * 70)
    print("QUESTION:", question)
    print("#" * 70)

    display_results("MiniLM (all-MiniLM-L6-v2)", minilm_results)
    display_results("BGE (BAAI/bge-small-en-v1.5)", bge_results)


if __name__ == "__main__":

    questions = [
        "What is the main idea of the Transformer architecture?",
        "What is the purpose of LoRA?",
        "What is Retrieval-Augmented Generation?",
        "How does BERT work?",
        "What is the purpose of GPT-3?"
    ]

    for question in questions:
        compare_models(question)