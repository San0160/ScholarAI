from abc import ABC, abstractmethod

from ai_research_assistant.entity.document import Document


class BaseVectorStore(ABC):

    @abstractmethod
    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]]
    ):
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[Document]:
        pass