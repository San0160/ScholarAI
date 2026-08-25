from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.evaluation.answer_evaluator import AnswerEvaluator


embedder = EmbeddingFactory.create_embedding()
evaluator = AnswerEvaluator(embedder)


question = (
    "What optimization algorithm and learning-rate schedule "
    "were used to train the Transformer?"
)

context = (
    "We used the Adam optimizer with β1 = 0.9, β2 = 0.98 and "
    "ε = 10−9. We varied the learning rate over the course of "
    "training, according to the formula: increasing the learning "
    "rate linearly for the first warmup_steps training steps, "
    "and decreasing it thereafter proportionally to the inverse "
    "square root of the step number."
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
    expected_answer=expected_answer,
    context=context
)


print("=" * 80)
print("ANSWER EVALUATION")
print("=" * 80)

print(f"Answer Relevance:    {results['answer_relevance']:.4f}")
print(f"Answer Correctness:  {results['answer_correctness']:.4f}")
print(f"Answer Groundedness: {results['answer_groundedness']:.4f}")