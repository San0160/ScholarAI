import numpy as np


class AnswerEvaluator:

    def __init__(
        self,
        embedder
    ):
        self.embedder = embedder

    def evaluate(
        self,
        question: str,
        generated_answer: str,
        expected_answer: str,
        context: str
    ) -> dict:

        if not generated_answer.strip():

            return {
                "answer_relevance": 0.0,
                "answer_correctness": 0.0,
                "answer_groundedness": 0.0
            }

        generated_embedding = np.array(
            self.embedder.embed_query(generated_answer)
        )

        expected_embedding = np.array(
            self.embedder.embed_query(expected_answer)
        )

        question_embedding = np.array(
            self.embedder.embed_query(question)
        )

        context_embedding = np.array(
            self.embedder.embed_query(
                context
            )
        )

        correctness = self._cosine_similarity(
            generated_embedding,
            expected_embedding
        )

        relevance = self._cosine_similarity(
            generated_embedding,
            question_embedding
        )

        groundedness = self._cosine_similarity(
            generated_embedding,
            context_embedding
        )

        return {
            "answer_relevance": float(relevance),
            "answer_correctness": float(correctness),
            "answer_groundedness": float(groundedness)
        }

    @staticmethod
    def _cosine_similarity(
        vector_a: np.ndarray,
        vector_b: np.ndarray
    ) -> float:

        norm_a = np.linalg.norm(
            vector_a
        )

        norm_b = np.linalg.norm(
            vector_b
        )

        if norm_a == 0 or norm_b == 0:

            return 0.0

        return float(
            np.dot(
                vector_a,
                vector_b
            )
            / (norm_a * norm_b)
        )