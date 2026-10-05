import sys
from pathlib import Path
import pandas as pd

# Add src folder to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))

from rag_pipeline_reranked import answer_question


TEST_FILE = PROJECT_ROOT / "evaluation" / "test_questions.csv"
OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "rag_evaluation_results.csv"


def check_paper_match(reranked_results, expected_paper):
    """
    Check whether the expected research paper
    appears in the top retrieved sources.
    """

    expected_paper = expected_paper.lower()

    retrieved_papers = []

    for document, score in reranked_results:
        source = Path(
            document.metadata.get("source", "")
        ).stem.lower()

        retrieved_papers.append(source)

    for paper in retrieved_papers:
        if expected_paper in paper or paper in expected_paper:
            return True

    return False


def main():

    print("=" * 70)
    print("RAG ANSWER EVALUATION")
    print("=" * 70)

    test_data = pd.read_csv(TEST_FILE)

    print(f"\nNumber of test questions: {len(test_data)}")

    results = []

    for index, row in test_data.iterrows():

        question = row["question"]
        expected_paper = row["expected_paper"]

        print("\n" + "=" * 70)
        print(f"QUESTION {index + 1}")
        print("=" * 70)

        print(f"Question: {question}")
        print(f"Expected paper: {expected_paper}")

        try:

            answer, reranked_results = answer_question(question)

            paper_match = check_paper_match(
                reranked_results,
                expected_paper
            )

            print("\nPaper retrieved correctly:", paper_match)

            results.append({
                "question": question,
                "expected_paper": expected_paper,
                "answer": answer,
                "correct_paper_retrieved": paper_match
            })

        except Exception as e:

            print("\nERROR:", e)

            results.append({
                "question": question,
                "expected_paper": expected_paper,
                "answer": "",
                "correct_paper_retrieved": False
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    accuracy = (
        results_df["correct_paper_retrieved"].mean()
        * 100
    )

    print("\n" + "=" * 70)
    print("RAG EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nCorrect paper retrieved: "
        f"{accuracy:.2f}%"
    )

    print(
        f"Results saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()