"""
Full evaluation runner: retrieval eval (RetrievalEvaluator) on both
chunk-level files, plus answer eval (AnswerEvaluator) on question_attention.json
specifically, since that's the only file with expected_answer.

question_multi_doc.json is intentionally excluded -- document-level
(expected_source), needs a separate evaluator.
"""

from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path

from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.pipeline.generation_pipeline import GenerationPipeline
from ai_research_assistant.evaluation.retrieval_evaluator import RetrievalEvaluator
from ai_research_assistant.evaluation.answer_evaluator import AnswerEvaluator

EVAL_FILES = {
    "GAN_case_study.pdf": Path("src/ai_research_assistant/evaluation/questions.json"),
    "attention.pdf": Path("src/ai_research_assistant/evaluation/question_attention.json"),
}

TOP_K = 3


def load_questions(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    config_manager = ConfigurationManager()
    embedder = EmbeddingFactory.create_embedding(config_manager.get_embedding_config())

    retrieval_pipeline = RetrievalPipeline()
    generation_pipeline = GenerationPipeline(embedder=embedder)

    retrieval_evaluator = RetrievalEvaluator(pipeline=retrieval_pipeline, k=TOP_K)
    answer_evaluator = AnswerEvaluator(embedder=embedder)

    for label, path in EVAL_FILES.items():
        questions = load_questions(path)

        print(f"=== Retrieval eval: {label} ({len(questions)} questions) ===")
        retrieval_results = retrieval_evaluator.evaluate(questions)
        print(f"  recall:    {retrieval_results['recall']:.3f}")
        print(f"  precision: {retrieval_results['precision']:.3f}")
        print(f"  mrr:       {retrieval_results['mrr']:.3f}")
        for detail in retrieval_results["details"]:
            status = "HIT" if detail["hit"] else "MISS"
            print(f"  [{status}] {detail['question']}")
            print(f"        expected:  {detail['expected_chunks']}")
            print(f"        retrieved: {detail['retrieved_chunks']}")
        print()

        has_expected_answer = any(q.get("expected_answer") for q in questions)
        if not has_expected_answer:
            print(f"(skipping answer eval for {label} -- no expected_answer field)\n")
            continue

        print(f"=== Answer eval: {label} ===")
        answer_scores = []
        answer_debug_records = []

        for question in questions:
            expected_answer = question.get("expected_answer")
            if not expected_answer:
                continue

            query = question["question"]
            retrieved = retrieval_pipeline.run(query)
            documents = [result.document for result in retrieved]

            generation_result = generation_pipeline.run(query, documents)
            generated_answer = generation_result["raw_answer"]
            context = generation_result["context"]

            scores = answer_evaluator.evaluate(
                question=query,
                generated_answer=generated_answer,
                expected_answer=expected_answer,
                context=context,
            )
            answer_scores.append(scores)
            answer_debug_records.append({
                "question": query,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "context": context,
                "scores": scores,
            })

            print(f"  Q: {query}")
            print(f"    relevance:    {scores['answer_relevance']:.3f}")
            print(f"    correctness:  {scores['answer_correctness']:.3f}")
            print(f"    groundedness: {scores['answer_groundedness']:.3f}")

        if answer_scores:
            avg_relevance = sum(s["answer_relevance"] for s in answer_scores) / len(answer_scores)
            avg_correctness = sum(s["answer_correctness"] for s in answer_scores) / len(answer_scores)
            avg_groundedness = sum(s["answer_groundedness"] for s in answer_scores) / len(answer_scores)
            print(f"\n  AVG relevance:    {avg_relevance:.3f}")
            print(f"  AVG correctness:  {avg_correctness:.3f}")
            print(f"  AVG groundedness: {avg_groundedness:.3f}")

            # Sanity check: dump the lowest-groundedness question in full
            lowest = min(answer_debug_records, key=lambda r: r["scores"]["answer_groundedness"])
            print("\n--- Lowest groundedness sanity check ---")
            print(f"Question: {lowest['question']}")
            print(f"Groundedness score: {lowest['scores']['answer_groundedness']:.3f}")
            print(f"\nExpected answer:\n{lowest['expected_answer']}")
            print(f"\nGenerated answer (raw):\n{lowest['generated_answer']}")
            print(f"\nContext used for generation:\n{lowest['context']}")
            print("--- end sanity check ---")
        print()


if __name__ == "__main__":
    main()