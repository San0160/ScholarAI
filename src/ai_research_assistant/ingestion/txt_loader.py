from pathlib import Path

from ai_research_assistant.entity.document import Document
from ai_research_assistant.ingestion.base_loader import BaseLoader


class TxtLoader(BaseLoader):

    def load(self, file_path: str) -> list[Document]:

        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as file:

            text = file.read().strip()

        if not text:
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_type": "txt"
                }
            )
        ]