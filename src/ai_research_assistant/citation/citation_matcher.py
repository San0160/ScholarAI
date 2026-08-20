import numpy as np


class CitationMatcher:

    def __init__(
        self,
        embedder,
        threshold: float = 0.65
    ):

        self.embedder = embedder
        self.threshold = threshold

    def match(
        self,
        answer: str,
        documents,
        source_map: dict
    ) -> list[dict]:

        if not answer.strip():
            return []

        answer_embedding = np.array(
            self.embedder.embed_query(
                answer
            )
        )

        matches = []

        for source_id, document in enumerate(
            documents,
            start=1
        ):

            document_embedding = np.array(
                self.embedder.embed_query(
                    document.page_content
                )
            )

            score = self._cosine_similarity(
                answer_embedding,
                document_embedding
            )

            matches.append(
                {
                    "source_id": source_id,
                    "score": float(score)
                }
            )

        if not matches:
            return []

        matches.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        best_match = matches[0]

        if best_match["score"] < self.threshold:
            return []

        return [best_match]

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
            / (
                norm_a * norm_b
            )
        )