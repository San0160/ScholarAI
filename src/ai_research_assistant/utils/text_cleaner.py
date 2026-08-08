import re

from ai_research_assistant.entity.document import Document


class TextCleaner:

    @staticmethod
    def clean(document: Document) -> Document:
        """
        Clean the text content of a Document.
        """

        text = document.page_content

        # Remove tabs
        text = text.replace("\t", " ")

        # Normalize multiple newlines
        text = re.sub(r"\n+", "\n", text)

        # Normalize multiple spaces
        text = re.sub(r" +", " ", text)

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
    