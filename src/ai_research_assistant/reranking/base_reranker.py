from abc import ABC, abstractmethod

from ai_research_assistant.entity.retrieval_result import RetrievalResult


class BaseReranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int
    ) -> list[RetrievalResult]:
        pass