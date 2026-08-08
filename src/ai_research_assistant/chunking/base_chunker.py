from abc import ABC, abstractmethod
from ai_research_assistant.entity.document import Document


class BaseChunker(ABC):

    @abstractmethod
    def split_documents(self, documents: list[Document]) -> list[Document]:
        pass