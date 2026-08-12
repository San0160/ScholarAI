from pathlib import Path
import re

import fitz

from ai_research_assistant.entity.document import Document


class PdfLoader:

    def load(
        self,
        file_path: str
    ) -> list[Document]:

        file_path = Path(file_path)

        documents = []

        current_section = None

        pdf = fitz.open(file_path)

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text().strip()

            if not text:
                continue

            detected_section = self._detect_heading(text)

            if detected_section:
                current_section = detected_section

            metadata = {
                "filename": file_path.name,
                "file_type": "pdf",
                "page": page_number
            }

            if current_section:
                metadata["section"] = current_section

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )

        pdf.close()

        return documents

    def _detect_heading(
        self,
        text: str
    ) -> str | None:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for line in lines:

            if len(line) > 80:
                continue

            lower_line = line.lower()

            # Ignore figure/table captions
            if lower_line.startswith(
                ("figure ", "fig. ", "table ")
            ):
                continue

            # Ignore lines that look like normal sentences
            if line.endswith("."):
                continue

            # Ignore lines with too many words
            if len(line.split()) > 8:
                continue

            # Numbered heading
            numbered_heading = re.match(
                r"^\d+(?:\.\d+)*\.?\s+(.+)$",
                line
            )

            if numbered_heading:

                heading = numbered_heading.group(1).strip()

                if len(heading.split()) <= 8:
                    return heading.lower()

            # Standalone heading
            if line[0].isupper():

                # Avoid headings containing obvious metadata
                if any(
                    token in lower_line
                    for token in [
                        "github",
                        "student",
                        "student id"
                    ]
                ):
                    continue

                return lower_line

        return None