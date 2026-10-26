from ai_research_assistant.entity.retrieval_result import RetrievalResult
from ai_research_assistant.reranking.base_reranker import BaseReranker


class NoReranker(BaseReranker):
    """Keeps the vector-search order and simply cuts to top_k. Used as the
    baseline when measuring how much the real reranker improves retrieval."""

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int
    ) -> list[RetrievalResult]:
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]
