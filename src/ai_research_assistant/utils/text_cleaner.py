import re

from ai_research_assistant.entity.document import Document

# Reference markers left in ChatGPT exports, e.g. 【2†L162-L170】. They can wrap
# across a line break in extracted PDF text. Requiring the digit and dagger keeps
# ordinary 【 】 brackets in Chinese or Japanese text untouched. A space left
# stranded before punctuation by the removal is taken out with the marker.
_EXPORT_MARKER_PATTERN = re.compile(
    r"[^\S\n]*【\d+†[^】]{0,40}】(?:[^\S\n]*(?=[.,;:]))?"
)


class TextCleaner:

    @staticmethod
    def clean(document: Document) -> Document:
        """
        Clean the text content of a Document.
        """

        text = document.page_content

        # Remove export reference markers (before whitespace is normalized)
        text = _EXPORT_MARKER_PATTERN.sub("", text)

        # Turn every kind of horizontal whitespace (tabs, non-breaking and thin
        # spaces, carriage returns) into a single ordinary space
        text = re.sub(r"[^\S\n]+", " ", text)

        # Drop spaces around line breaks, then normalize multiple newlines
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove non-printable characters
        text = "".join(char for char in text if char.isprintable() or char == "\n")

        # Strip leading/trailing whitespace
        text = text.strip()

        return Document(
            page_content=text,
            metadata=document.metadata.copy()
        )

    @staticmethod
    def clean_documents(documents: list[Document]) -> list[Document]:
        """
        Clean a list of Documents.
        """

        return [
            TextCleaner.clean(document)
            for document in documents
        ]