from abc import ABC, abstractmethod
from ai_research_assistant.entity.document import Document


class BaseLoader(ABC):

    @abstractmethod
    def load(self, file_path: str) -> list[Document]:
        """
        Load a document and return a list of Document objects.
        """
        pass