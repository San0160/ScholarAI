from abc import ABC, abstractmethod

from ai_research_assistant.entity.document import Document


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> list[Document]:
        pass