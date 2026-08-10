from abc import ABC, abstractmethod


class BaseEmbedding(ABC):

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        pass

    @abstractmethod
    def embed_query(
        self,
        text: str
    ) -> list[float]:
        pass