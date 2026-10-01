from pathlib import Path
from ai_research_assistant.ingestion.txt_loader import TxtLoader
from ai_research_assistant.ingestion.pdf_loader import PdfLoader
from ai_research_assistant.ingestion.docx_loader import DocxLoader
from ai_research_assistant.ingestion.markdown_loader import MarkdownLoader

class DocumentLoader:

    def __init__(self):

        self.loaders = {
            ".txt": TxtLoader(),
            ".pdf": PdfLoader(),
            ".docx": DocxLoader(),
            ".md": MarkdownLoader(),
        }

    def load(self, file_path: str):
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        extension = path.suffix.lower()

        if extension not in self.loaders:
            raise ValueError(
                f"Unsupported file type: {extension}. Allowed: {sorted(self.loaders)}"
            )

        return self.loaders[extension].load(file_path)