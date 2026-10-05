from pathlib import Path
import csv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


MINILM_CHROMA_FOLDER = "results/chroma_db"
BGE_CHROMA_FOLDER = "results/chroma_db_bge"


def load_minilm_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=MINILM_CHROMA_FOLDER,
        collection_name="research_papers",
        embedding_function=embeddings
    )


def load_bge_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return Chroma(
        persist_directory=BGE_CHROMA_FOLDER,
        collection_name="research_papers_bge",
        embedding_function=embeddings
    )


def get_paper_name(document):
    source = document.metadata.get("source", "Unknown")
    return Path(source).stem.lower()


def evaluate_model(vectorstore, questions):
    results = []

    top1_correct = 0
    top3_correct = 0

    for item in questions:

        question = item["question"]
        expected_paper = item["expected_paper"].lower()

        retrieved = vectorstore.similarity_search_with_score(
            question,
            k=3
        )

        retrieved_papers = [
            get_paper_name(document)
            for document, score in retrieved
        ]

        top1_match = (
            expected_paper == retrieved_papers[0]
        )

        top3_match = (
            expected_paper in retrieved_papers
        )

        if top1_match:
            top1_correct += 1

        if top3_match:
            top3_correct += 1

        results.append({
            "question": question,
            "expected_paper": expected_paper,
            "retrieved_papers": " | ".join(retrieved_papers),
            "top1_correct": top1_match,
            "top3_correct": top3_match
        })

    total = len(questions)

    top1_accuracy = (top1_correct / total) * 100
    top3_accuracy = (top3_correct / total) * 100

    return results, top1_accuracy, top3_accuracy


def load_questions():

    questions = []

    with open(
        "evaluation/test_questions.csv",
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            questions.append(row)

    return questions


def save_results(model_name, results, top1_accuracy, top3_accuracy):

    output_file = "evaluation/embedding_results.csv"

    file_exists = Path(output_file).exists()

    with open(
        output_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "model",
                "question",
                "expected_paper",
                "retrieved_papers",
                "top1_correct",
                "top3_correct",
                "top1_accuracy",
                "top3_accuracy"
            ])

        for result in results:

            writer.writerow([
                model_name,
                result["question"],
                result["expected_paper"],
                result["retrieved_papers"],
                result["top1_correct"],
                result["top3_correct"],
                f"{top1_accuracy:.2f}",
                f"{top3_accuracy:.2f}"
            ])


def main():

    print("=" * 70)
    print("EMBEDDING EVALUATION")
    print("=" * 70)

    questions = load_questions()

    print(f"\nNumber of test questions: {len(questions)}")

    print("\nLoading MiniLM database...")
    minilm_db = load_minilm_database()

    print("Loading BGE database...")
    bge_db = load_bge_database()

    print("\nEvaluating MiniLM...")
    
    minilm_results, minilm_top1, minilm_top3 = evaluate_model(
        minilm_db,
        questions
    )

    print("\nEvaluating BGE...")

    bge_results, bge_top1, bge_top3 = evaluate_model(
        bge_db,
        questions
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"\nMiniLM Top-1 Accuracy : {minilm_top1:.2f}%"
    )

    print(
        f"MiniLM Top-3 Accuracy : {minilm_top3:.2f}%"
    )

    print(
        f"\nBGE Top-1 Accuracy : {bge_top1:.2f}%"
    )

    print(
        f"BGE Top-3 Accuracy : {bge_top3:.2f}%"
    )

    save_results(
        "MiniLM",
        minilm_results,
        minilm_top1,
        minilm_top3
    )

    save_results(
        "BGE",
        bge_results,
        bge_top1,
        bge_top3
    )

    print(
        "\nResults saved to: evaluation/embedding_results.csv"
    )


if __name__ == "__main__":
    main()