from sentence_transformers import CrossEncoder

from ai_research_assistant.entity.retrieval_result import RetrievalResult
from ai_research_assistant.reranking.base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):

    def __init__(self, model_name: str):

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int
    ) -> list[RetrievalResult]:

        pairs = [
            (query, result.document.page_content)
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for result, score in zip(results, scores):

            reranked_results.append(
                RetrievalResult(
                    document=result.document,
                    score=float(score)
                )
            )

        reranked_results.sort(
            key=lambda result: result.score,
            reverse=True
        )

        return reranked_results[:top_k]