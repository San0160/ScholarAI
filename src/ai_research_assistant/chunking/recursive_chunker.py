# Logic, if paragrarph is too large convert to sentences and if sentence is too large then chunk to words

from ai_research_assistant.entity.document import Document
from ai_research_assistant.chunking.base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(
        self,
        documents: list[Document]
    ) -> list[Document]:

        chunks = []

        for document in documents:

            text_chunks = self._split_text(
                document.page_content
            )

            for chunk in text_chunks:

                if not chunk.strip():
                    continue

                metadata = document.metadata.copy()

                metadata["chunk_id"] = (
                    f"{metadata.get('filename', 'document')}_"
                    f"{len(chunks)}"
                )

                chunks.append(
                    Document(
                        page_content=chunk.strip(),
                        metadata=metadata
                    )
                )

        return chunks

    def _split_text(
        self,
        text: str
    ) -> list[str]:

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:

            # Paragraph itself fits
            if len(paragraph) <= self.chunk_size:

                candidate = (
                    f"{current_chunk}\n\n{paragraph}"
                    if current_chunk
                    else paragraph
                )

                if len(candidate) <= self.chunk_size:

                    current_chunk = candidate

                else:

                    if current_chunk:
                        chunks.append(current_chunk)

                    current_chunk = paragraph

            # Paragraph is larger than chunk size
            else:

                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                chunks.extend(
                    self._split_large_paragraph(paragraph)
                )

        if current_chunk:
            chunks.append(current_chunk)

        return self._apply_overlap(chunks)

    def _split_large_paragraph(
        self,
        text: str
    ) -> list[str]:

        sentences = text.split(". ")

        chunks = []
        current_chunk = ""

        for sentence in sentences:

            if not sentence.strip():
                continue

            sentence = sentence.strip()

            if not sentence.endswith("."):
                sentence += "."

            candidate = (
                f"{current_chunk} {sentence}"
                if current_chunk
                else sentence
            )

            if len(candidate) <= self.chunk_size:

                current_chunk = candidate

            else:

                if current_chunk:
                    chunks.append(current_chunk)

                # Single sentence larger than chunk size
                if len(sentence) > self.chunk_size:

                    chunks.extend(
                        self._split_by_words(sentence)
                    )

                    current_chunk = ""

                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_by_words(
        self,
        text: str
    ) -> list[str]:

        words = text.split()

        chunks = []
        current_chunk = ""

        for word in words:

            candidate = (
                f"{current_chunk} {word}"
                if current_chunk
                else word
            )

            if len(candidate) <= self.chunk_size:

                current_chunk = candidate

            else:

                if current_chunk:
                    chunks.append(current_chunk)

                current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _apply_overlap(
        self,
        chunks: list[str]
    ) -> list[str]:

        if self.chunk_overlap <= 0:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):

            previous = chunks[i - 1]

            overlap = previous[-self.chunk_overlap:]

            combined = (
                overlap + "\n\n" + chunks[i]
            )

            # Prevent overlap from making chunk too large
            if len(combined) > self.chunk_size:

                combined = chunks[i]

            result.append(combined)

        return result