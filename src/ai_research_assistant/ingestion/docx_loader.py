from pathlib import Path
from docx import Document as DocxDocument

from ai_research_assistant.entity.document import Document
from ai_research_assistant.ingestion.base_loader import BaseLoader


class DocxLoader(BaseLoader):

    def load(self, file_path: str) -> list[Document]:

        doc = DocxDocument(file_path)

        text = "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        )

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "filename": Path(file_path).name,
                    "file_type": "docx"
                }
            )
        ]