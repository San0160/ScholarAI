from ai_research_assistant.entity.document import Document
from ai_research_assistant.ingestion.base_loader import BaseLoader
import fitz
from pathlib import Path


class PDFLoader(BaseLoader):

    def load(self, file_path: str) -> list[Document]:

        documents = []

        pdf = fitz.open(file_path)

        for page_number, page in enumerate(pdf, start=1):

            documents.append(
                Document(
                    page_content=page.get_text(),
                    metadata={
                        "source": file_path,
                        "filename": Path(file_path).name,
                        "file_type": "pdf",
                        "page": page_number
                    }
                )
            )

        pdf.close()

        return documents