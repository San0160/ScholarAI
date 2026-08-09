from abc import ABC, abstractmethod


class BaseEmbedding(ABC):

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a user query.
        """
        pass