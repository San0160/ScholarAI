from abc import ABC, abstractmethod

from ai_research_assistant.entity.retrieval_result import RetrievalResult

class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> list[RetrievalResult]:
        pass