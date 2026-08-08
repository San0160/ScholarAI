from ai_research_assistant.entity.document import Document
from ai_research_assistant.ingestion.base_loader import BaseLoader
from pathlib import Path


class TxtLoader(BaseLoader):

    def load(self, file_path: str) -> list[Document]:

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "filename": Path(file_path).name,
                    "file_type": "txt"
                }
            )
        ]