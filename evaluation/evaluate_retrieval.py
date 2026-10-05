from pathlib import Path
import sys
import csv

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from chunk_papers import load_all_papers, split_documents
from hybrid_retrieval import create_bm25_index, hybrid_retrieve
from reranking import load_vector_database, rerank_results


TEST_FILE = "evaluation/test_questions.csv"


def get_paper_name(document):
    source = document.metadata.get("source", "")
    return Path(source).stem.lower()


def evaluate_top_k(results, expected_paper, k):
    retrieved_papers = [
        get_paper_name(document)
        for document in results[:k]
    ]

    return expected_paper.lower() in retrieved_papers


def get_bm25_results(question, chunks, bm25, k=10):
    tokenized_question = question.lower().split()

    bm25_scores = bm25.get_scores(tokenized_question)

    ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:k]

    return [
        chunks[i]
        for i in ranked_indices
    ]


def main():

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)

    print("\nLoading research papers...")

    documents = load_all_papers()
    chunks = split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    print("\nLoading BGE vector database...")

    vectorstore = load_vector_database()

    print("\nCreating BM25 index...")

    bm25 = create_bm25_index(chunks)

    print("\nLoading test questions...")

    test_questions = []

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            test_questions.append({
                "question": row["question"],
                "expected_paper": row["expected_paper"]
            })

    print(f"Number of test questions: {len(test_questions)}")

    dense_top1 = 0
    dense_top3 = 0

    bm25_top1 = 0
    bm25_top3 = 0

    hybrid_top1 = 0
    hybrid_top3 = 0

    rerank_top1 = 0
    rerank_top3 = 0

    for index, item in enumerate(test_questions, start=1):

        question = item["question"]
        expected = item["expected_paper"]

        print("\n" + "-" * 70)
        print(f"Question {index}: {question}")
        print(f"Expected paper: {expected}")

        dense_results = vectorstore.similarity_search(
            question,
            k=10
        )

        dense_1 = evaluate_top_k(
            dense_results,
            expected,
            1
        )

        dense_3 = evaluate_top_k(
            dense_results,
            expected,
            3
        )

        if dense_1:
            dense_top1 += 1

        if dense_3:
            dense_top3 += 1

        bm25_results = get_bm25_results(
            question,
            chunks,
            bm25,
            k=10
        )

        bm25_1 = evaluate_top_k(
            bm25_results,
            expected,
            1
        )

        bm25_3 = evaluate_top_k(
            bm25_results,
            expected,
            3
        )

        if bm25_1:
            bm25_top1 += 1

        if bm25_3:
            bm25_top3 += 1

        hybrid_results = hybrid_retrieve(
            question=question,
            vectorstore=vectorstore,
            bm25=bm25,
            documents=chunks,
            k=10
        )

        hybrid_1 = evaluate_top_k(
            hybrid_results,
            expected,
            1
        )

        hybrid_3 = evaluate_top_k(
            hybrid_results,
            expected,
            3
        )

        if hybrid_1:
            hybrid_top1 += 1

        if hybrid_3:
            hybrid_top3 += 1

        reranked_results = rerank_results(
            question,
            hybrid_results,
            top_k=10
        )

        reranked_documents = [
            document
            for document, score in reranked_results
        ]

        rerank_1 = evaluate_top_k(
            reranked_documents,
            expected,
            1
        )

        rerank_3 = evaluate_top_k(
            reranked_documents,
            expected,
            3
        )

        if rerank_1:
            rerank_top1 += 1

        if rerank_3:
            rerank_top3 += 1

        print(
            f"Dense BGE       → "
            f"Top-1: {'✓' if dense_1 else '✗'} | "
            f"Top-3: {'✓' if dense_3 else '✗'}"
        )

        print(
            f"BM25            → "
            f"Top-1: {'✓' if bm25_1 else '✗'} | "
            f"Top-3: {'✓' if bm25_3 else '✗'}"
        )

        print(
            f"Hybrid          → "
            f"Top-1: {'✓' if hybrid_1 else '✗'} | "
            f"Top-3: {'✓' if hybrid_3 else '✗'}"
        )

        print(
            f"Hybrid + Rerank → "
            f"Top-1: {'✓' if rerank_1 else '✗'} | "
            f"Top-3: {'✓' if rerank_3 else '✗'}"
        )

    total = len(test_questions)

    dense_top1_accuracy = dense_top1 / total * 100
    dense_top3_accuracy = dense_top3 / total * 100

    bm25_top1_accuracy = bm25_top1 / total * 100
    bm25_top3_accuracy = bm25_top3 / total * 100

    hybrid_top1_accuracy = hybrid_top1 / total * 100
    hybrid_top3_accuracy = hybrid_top3 / total * 100

    rerank_top1_accuracy = rerank_top1 / total * 100
    rerank_top3_accuracy = rerank_top3 / total * 100

    print("\n" + "=" * 70)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"\nDense BGE       → "
        f"Top-1: {dense_top1_accuracy:.2f}% | "
        f"Top-3: {dense_top3_accuracy:.2f}%"
    )

    print(
        f"BM25            → "
        f"Top-1: {bm25_top1_accuracy:.2f}% | "
        f"Top-3: {bm25_top3_accuracy:.2f}%"
    )

    print(
        f"Hybrid          → "
        f"Top-1: {hybrid_top1_accuracy:.2f}% | "
        f"Top-3: {hybrid_top3_accuracy:.2f}%"
    )

    print(
        f"Hybrid + Rerank → "
        f"Top-1: {rerank_top1_accuracy:.2f}% | "
        f"Top-3: {rerank_top3_accuracy:.2f}%"
    )

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()