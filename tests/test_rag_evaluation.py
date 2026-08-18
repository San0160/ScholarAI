import json

from ai_research_assistant.embeddings.embedding_factory import (
    EmbeddingFactory
)

from ai_research_assistant.pipeline.retrieval_pipeline import (
    RetrievalPipeline
)

from ai_research_assistant.pipeline.generation_pipeline import (
    GenerationPipeline
)

from ai_research_assistant.generation.context_builder import (
    ContextBuilder
)

from ai_research_assistant.evaluation.answer_evaluator import (
    AnswerEvaluator
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

QUESTION_FILE = "src/ai_research_assistant/evaluation/question_attention.json"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# --------------------------------------------------
# Load Questions
# --------------------------------------------------

with open(
    QUESTION_FILE,
    "r",
    encoding="utf-8"
) as file:

    questions = json.load(file)


# --------------------------------------------------
# Create Embedding Model
# --------------------------------------------------

embedder = EmbeddingFactory.create_embedding()


# --------------------------------------------------
# Initialize Components
# --------------------------------------------------

retrieval_pipeline = RetrievalPipeline()

generation_pipeline = GenerationPipeline(
    model_name=MODEL_NAME
)

context_builder = ContextBuilder()

answer_evaluator = AnswerEvaluator(
    embedder=embedder
)


# --------------------------------------------------
# Evaluation Storage
# --------------------------------------------------

results = []


# --------------------------------------------------
# Run End-to-End Evaluation
# --------------------------------------------------

for index, item in enumerate(
    questions,
    start=1
):

    question = item["question"]

    expected_answer = item.get(
        "expected_answer",
        ""
    )

    expected_chunks = item.get(
        "expected_chunks",
        []
    )


    # ----------------------------------------------
    # Retrieval
    # ----------------------------------------------

    retrieved_results = retrieval_pipeline.run(
        question
    )


    retrieved_chunks = [
        result.document.metadata.get(
            "chunk_id"
        )
        for result in retrieved_results
    ]


    documents = [
        result.document
        for result in retrieved_results
    ]


    # ----------------------------------------------
    # Retrieval Hit
    # ----------------------------------------------

    retrieval_hit = any(
        chunk in retrieved_chunks
        for chunk in expected_chunks
    )


    # ----------------------------------------------
    # Build Context
    # ----------------------------------------------

    context = context_builder.build(
        documents
    )


    # ----------------------------------------------
    # Generate Answer
    # ----------------------------------------------

    generated_answer = generation_pipeline.run(
        query=question,
        documents=documents
    )


    # ----------------------------------------------
    # Evaluate Answer
    # ----------------------------------------------

    answer_scores = answer_evaluator.evaluate(
        question=question,
        generated_answer=generated_answer,
        expected_answer=expected_answer,
        context=context
    )


    # ----------------------------------------------
    # Store Results
    # ----------------------------------------------

    results.append(
        {
            "question": question,
            "retrieval_hit": retrieval_hit,
            "retrieved_chunks": retrieved_chunks,
            "generated_answer": generated_answer,
            "answer_relevance": answer_scores[
                "answer_relevance"
            ],
            "answer_correctness": answer_scores[
                "answer_correctness"
            ],
            "answer_groundedness": answer_scores[
                "answer_groundedness"
            ]
        }
    )


    # ----------------------------------------------
    # Compact Output
    # ----------------------------------------------

    print("\n" + "=" * 80)
    print(f"QUESTION {index}")
    print("=" * 80)

    print(
        f"Question: {question}"
    )

    print(
        f"Retrieval Hit: {retrieval_hit}"
    )

    print(
        f"Retrieved Chunks: {retrieved_chunks}"
    )

    print(
        f"Answer Relevance: "
        f"{answer_scores['answer_relevance']:.4f}"
    )

    print(
        f"Answer Correctness: "
        f"{answer_scores['answer_correctness']:.4f}"
    )

    print(
        f"Answer Groundedness: "
        f"{answer_scores['answer_groundedness']:.4f}"
    )

    print(
        f"Generated Answer: "
        f"{generated_answer}"
    )


# --------------------------------------------------
# Overall Metrics
# --------------------------------------------------

total_questions = len(
    results
)


retrieval_hits = sum(
    result["retrieval_hit"]
    for result in results
)


average_relevance = sum(
    result["answer_relevance"]
    for result in results
) / total_questions


average_correctness = sum(
    result["answer_correctness"]
    for result in results
) / total_questions


average_groundedness = sum(
    result["answer_groundedness"]
    for result in results
) / total_questions


# --------------------------------------------------
# Final Summary
# --------------------------------------------------

print("\n" + "=" * 80)
print("END-TO-END RAG EVALUATION")
print("=" * 80)

print(
    f"Retrieval Hit Rate: "
    f"{retrieval_hits / total_questions:.4f}"
)

print(
    f"Average Answer Relevance: "
    f"{average_relevance:.4f}"
)

print(
    f"Average Answer Correctness: "
    f"{average_correctness:.4f}"
)

print(
    f"Average Answer Groundedness: "
    f"{average_groundedness:.4f}"
)