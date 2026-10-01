import sys
from pathlib import Path

from ai_research_assistant.entity.document import Document
from ai_research_assistant.exception import CustomException
from ai_research_assistant.ingestion.base_loader import BaseLoader


class TxtLoader(BaseLoader):

    def load(self, file_path: str) -> list[Document]:
        file_path = Path(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read().strip()
        except Exception as e:
            raise CustomException(e, sys) from e

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