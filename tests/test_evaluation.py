import json

from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.evaluation.retrival_evaluator import RetrievalEvaluator


QUESTION_FILE = (
    "src/ai_research_assistant/evaluation/"
    "question_attention.json"
)


def main():

    # --------------------------------------------------
    # Load evaluation questions
    # --------------------------------------------------

    with open(
        QUESTION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        questions = json.load(file)

    # --------------------------------------------------
    # Retrieval Pipeline
    # --------------------------------------------------

    pipeline = RetrievalPipeline()

    # --------------------------------------------------
    # Evaluator
    # --------------------------------------------------

    evaluator = RetrievalEvaluator(
        pipeline=pipeline,
        k=3
    )

    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------

    result = evaluator.evaluate(
        questions
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print(
        f"Retrieval Recall@3: "
        f"{result['recall']}"
    )

    print(
        f"Retrieval Precision@3: "
        f"{result['precision']}"
    )

    print(
        f"MRR@3: "
        f"{result['mrr']}"
    )

    print(
        "\nDetailed Results:"
    )

    for detail in result["details"]:

        print(
            f"\nQuestion: "
            f"{detail['question']}"
        )

        print(
            f"Expected chunks: "
            f"{detail['expected_chunks']}"
        )

        print(
            f"Retrieved chunks: "
            f"{detail['retrieved_chunks']}"
        )

        print(
            f"Hit: "
            f"{detail['hit']}"
        )


if __name__ == "__main__":
    main()