"""
Evaluation runner, retrieval half: runs RetrievalEvaluator against both
chunk-level ground truth files using the live RetrievalPipeline.

question_multi_doc.json is intentionally excluded -- it's document-level
(expected_source), not chunk-level, and needs a separate evaluator.
"""

from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path

from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.evaluation.retrieval_evaluator import RetrievalEvaluator

EVAL_FILES = {
    "GAN_case_study.pdf": Path("src/ai_research_assistant/evaluation/questions.json"),
    "attention.pdf": Path("src/ai_research_assistant/evaluation/question_attention.json"),
}

TOP_K = 3


def load_questions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    retrieval_pipeline = RetrievalPipeline()
    retrieval_evaluator = RetrievalEvaluator(pipeline=retrieval_pipeline, k=TOP_K)

    for label, path in EVAL_FILES.items():
        questions = load_questions(path)
        print(f"=== Retrieval eval: {label} ({len(questions)} questions) ===")
        results = retrieval_evaluator.evaluate(questions)
        print(f"  recall:    {results['recall']:.3f}")
        print(f"  precision: {results['precision']:.3f}")
        print(f"  mrr:       {results['mrr']:.3f}")
        for detail in results["details"]:
            status = "HIT" if detail["hit"] else "MISS"
            print(f"  [{status}] {detail['question']}")
            print(f"        expected:  {detail['expected_chunks']}")
            print(f"        retrieved: {detail['retrieved_chunks']}")
        print()


if __name__ == "__main__":
    main()