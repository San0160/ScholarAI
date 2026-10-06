from FlagEmbedding import FlagReranker

from ai_research_assistant.entity.retrieval_result import RetrievalResult
from ai_research_assistant.reranking.base_reranker import BaseReranker


class BGEReranker(BaseReranker):

    def __init__(
        self,
        model_name: str
    ):

        self.model = FlagReranker(
            model_name,
            use_fp16=True
        )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int
    ) -> list[RetrievalResult]:

        pairs = [
            [
                query,
                result.document.page_content
            ]
            for result in results
        ]

        scores = self.model.compute_score(
            pairs,
            normalize=True
        )

        reranked_results = []

        for result, score in zip(
            results,
            scores
        ):

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