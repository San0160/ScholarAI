import numpy as np


class CitationMatcher:

    def __init__(self, embedder, threshold: float = 0.65, max_citations: int = 3):
        self.embedder = embedder
        self.threshold = threshold
        self.max_citations = max_citations

    def match(self, answer: str, documents, source_map: dict) -> list[dict]:
        if not answer.strip():
            return []
        if not documents:
            return []

        answer_embedding = np.array(self.embedder.embed_query(answer))
        document_embeddings = np.array(
            self.embedder.embed_documents([document.page_content for document in documents])
        )

        scores = self._cosine_similarity_batch(answer_embedding, document_embeddings)

        matches = [
            {"source_id": source_id, "score": float(score)}
            for source_id, score in enumerate(scores, start=1)
            if score >= self.threshold
        ]

        matches.sort(key=lambda item: item["score"], reverse=True)

        return matches[: self.max_citations]

    @staticmethod
    def _cosine_similarity_batch(query_vector: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query_vector)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        denom = query_norm * matrix_norms
        safe_denom = np.where(denom == 0, 1.0, denom)
        similarities = (matrix @ query_vector) / safe_denom
        return np.where(denom == 0, 0.0, similarities)