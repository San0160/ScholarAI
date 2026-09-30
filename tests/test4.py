from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path

from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.evaluation.document_retrieval_evaluator import DocumentRetrievalEvaluator

QUESTIONS_FILE = Path("src/ai_research_assistant/evaluation/question_multi_doc.json")
STORAGE_PATH = "artifacts/vector_store/multi_document"

TOP_K = 3


def main():
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    retrieval_pipeline = RetrievalPipeline(storage_path=STORAGE_PATH)
    evaluator = DocumentRetrievalEvaluator(pipeline=retrieval_pipeline, k=TOP_K)

    results = evaluator.evaluate(questions)

    print(f"=== Document-level retrieval eval ({len(questions)} questions) ===")
    print(f"  hit_rate:  {results['hit_rate']:.3f}")
    print(f"  precision: {results['precision']:.3f}")
    print(f"  mrr:       {results['mrr']:.3f}")
    for detail in results["details"]:
        status = "HIT" if detail["hit"] else "MISS"
        print(f"  [{status}] {detail['question']}")
        print(f"        expected: {detail['expected_source']}")
        print(f"        retrieved sources: {detail['retrieved_sources']}")


if __name__ == "__main__":
    main()