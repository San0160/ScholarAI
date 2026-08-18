from ai_research_assistant.evaluation.answer_evaluator import (
    AnswerEvaluator
)


evaluator = AnswerEvaluator()


question = (
    "What optimization algorithm and learning-rate schedule "
    "were used to train the Transformer?"
)


expected_answer = (
    "The Transformer was trained using the Adam optimizer "
    "with β1 = 0.9, β2 = 0.98, and ε = 10−9. "
    "The learning rate was increased linearly during the "
    "warmup period and then decreased proportionally to "
    "the inverse square root of the step number."
)


generated_answer = (
    "The optimization algorithm used was Adam with "
    "β1 = 0.9, β2 = 0.98, and ε = 10^-9. "
    "The learning rate increased linearly during "
    "the warmup steps and then decreased according "
    "to the inverse square root of the step number."
)


results = evaluator.evaluate(
    question=question,
    generated_answer=generated_answer,
    expected_answer=expected_answer
)


print("=" * 80)
print("ANSWER EVALUATION")
print("=" * 80)

print(
    f"Answer Relevance: "
    f"{results['answer_relevance']:.4f}"
)

print(
    f"Answer Correctness: "
    f"{results['answer_correctness']:.4f}"
)