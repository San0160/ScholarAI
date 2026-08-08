from ai_research_assistant.entity.document import Document
from ai_research_assistant.ingestion.base_loader import BaseLoader


class MarkdownLoader(BaseLoader):

    def load(self, file_path: str) -> list[Document]:

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "file_type": "txt"
                }
            )
        ]