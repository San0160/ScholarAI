from pathlib import Path
import re
from collections import Counter
import fitz

from ai_research_assistant.entity.document import Document


class PdfLoader:

    MIN_HEADING_FONT_RATIO = 1.15
    MAX_HEADING_WORDS = 8
    MAX_HEADING_CHARS = 80

    # Generic structural noise filters — not tied to any specific document's content
    NOISE_PATTERNS = [
        re.compile(r"https?://", re.IGNORECASE),
        re.compile(r"^www\.", re.IGNORECASE),
        re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),      # email addresses
        re.compile(r"^(figure|fig\.|table)\s", re.IGNORECASE),
        re.compile(r"^\d+$"),                          # standalone page numbers
    ]

    def load(self, file_path: str) -> list[Document]:      

        file_path = Path(file_path)
        pdf = fitz.open(file_path)

        body_font_size = self._estimate_body_font_size(pdf)

        documents = []
        current_section = None
        
        for page_number, page in enumerate(pdf, start=1):

            lines = self._extract_lines(page)

            if not lines:
                continue

            segments = self._segment_by_heading(
                lines,
                body_font_size,
                current_section
            )

            for section, text in segments:

                text = text.strip()

                if not text:
                    continue

                current_section = section

            metadata = {
                "filename": file_path.name,
                "file_type": "pdf",
                "page": page_number
            }

            if current_section:
                metadata["section"] = current_section

            documents.append(Document(page_content=text, metadata=metadata))

        pdf.close()

        return documents

    def _estimate_body_font_size(self, pdf) -> float:
        """
        Finds the most common font size across the document — this is
        almost always the body text size, since body text vastly
        outnumbers headings/captions in any real document.
        """

        sizes = []

        for page in pdf:
            raw = page.get_text("dict")
            for block in raw.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span["text"].strip():
                            sizes.append(round(span["size"], 1))

        if not sizes:
            return 10.0

        return Counter(sizes).most_common(1)[0][0]    

    def _extract_lines(self, page) -> list[dict]:
        """
        Lines extraction with font metadata
        """

        lines = []
        raw = page.get_text("dict")

        for block in raw.get("blocks", []):
            for line in block.get("lines", []):

                spans = line.get("spans", [])

                if not spans:
                    continue

                text = "".join(span["text"] for span in spans).strip()

                if not text:
                    continue

                max_size = max(span["size"] for span in spans)
                is_bold = any(
                    "bold" in span.get("font", "").lower()
                    for span in spans
                )

                lines.append({
                    "text": text,
                    "size": max_size,
                    "bold": is_bold
                })

        return lines

    def _is_heading(self, line: dict, body_size: float) -> bool:

        text = line["text"]

        if len(text) > self.MAX_HEADING_CHARS:
            return False

        if len(text.split()) > self.MAX_HEADING_WORDS:
            return False

        if text.endswith("."):
            return False

        for pattern in self.NOISE_PATTERNS:
            if pattern.search(text):
                return False

        size_ratio = (line["size"] / body_size) if body_size else 1.0

        looks_larger_or_bold = size_ratio >= self.MIN_HEADING_FONT_RATIO or line["bold"]

        if not looks_larger_or_bold:
            return False

        # Still require it to structurally resemble a heading
        # (numbered section, or starts with a capital letter)
        numbered = re.match(r"^\d+(?:\.\d+)*\.?\s+\S", text)

        return bool(numbered) or text[0].isupper()

    def _clean_heading_text(self, text: str) -> str:

        numbered_heading = re.match(
            r"^\d+(?:\.\d+)*\.?\s+(.+)$",
            text
        )

        if numbered_heading:
            return numbered_heading.group(1).strip().lower()

        return text.strip().lower()

    def _segment_by_heading(
        self,
        lines: list[dict],
        body_size: float,
        current_section: str | None
    ) -> list[tuple[str | None, str]]:

        segments = []
        buffer = []
        section = current_section

        for line in lines:

            if self._is_heading(line, body_size):

                if buffer:
                    segments.append((section, "\n".join(buffer)))
                    buffer = []

                section = self._clean_heading_text(line["text"])
                continue

            buffer.append(line["text"])

        if buffer:
            segments.append((section, "\n".join(buffer)))

        return segments