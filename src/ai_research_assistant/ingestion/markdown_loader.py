from pathlib import Path

import re

from ai_research_assistant.entity.document import Document
from ai_research_assistant.ingestion.base_loader import BaseLoader


class MarkdownLoader(BaseLoader):

    def load(self, file_path: str) -> list[Document]:

        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as file:

            text = file.read()

        documents = []
        current_section = None
        buffer = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            heading_match = re.match(r"^#{1,6}\s+(.+)$", line)

            if heading_match:
                if buffer:
                    documents.append(self._create_document("\n".join(buffer), file_path, current_section))

                    buffer = []

                current_section = (
                    heading_match.group(1)
                    .strip()
                    .lower()
                )

                continue

            buffer.append(line)

        if buffer:

            documents.append(
                self._create_document(
                    "\n".join(buffer),
                    file_path,
                    current_section
                )
            )

        return documents

    def _create_document(self, text: str, file_path: Path, section: str | None) -> Document:

        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": "md"
        }

        if section:
            metadata["section"] = section

        return Document(
            page_content=text,
            metadata=metadata
        )