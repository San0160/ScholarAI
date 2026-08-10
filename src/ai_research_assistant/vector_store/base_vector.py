from abc import ABC, abstractmethod
from ai_research_assistant.entity.retrieval_result import RetrievalResult

class BaseVectorStore(ABC):

    @abstractmethod
    def add_documents(
        self,
        documents,
        embeddings
    ):
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding,
        top_k = 5
    ) -> list[RetrievalResult]:
        pass