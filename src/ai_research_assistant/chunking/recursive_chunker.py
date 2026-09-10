from dataclasses import dataclass
from typing import Callable
import logging

from ai_research_assistant.entity.document import Document
from ai_research_assistant.chunking.base_chunker import BaseChunker

logger = logging.getLogger(__name__)

@dataclass
class _TextSpan:
    """A chunk's text plus its exact character offsets in the source
    document text. start/end always describe the chunk's OWN span --
    text borrowed from the previous chunk via overlap is not included
    in the range (see _apply_overlap)."""
    text: str
    start: int
    end: int


class RecursiveChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        token_counter: Callable[[str], int],
        sentence_splitter: Callable[[str], list[str]],
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size}), otherwise a chunk could be all overlap"
            )

        if not callable(token_counter):
            raise TypeError("token_counter must be callable")

        if not callable(sentence_splitter):
            raise TypeError("sentence_splitter must be callable")

        if chunk_overlap > chunk_size // 2:
            logger.warning(
                "chunk_overlap (%d) is more than half of chunk_size (%d) — "
                "chunks will be mostly duplicated content with limited retrieval benefit",
                chunk_overlap,
                chunk_size,
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.token_counter = token_counter
        self.sentence_splitter = sentence_splitter

    def _fits(self, text: str) -> bool:
        return self.token_counter(text) <= self.chunk_size

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks = []

        for document in documents:
            spans = self._split_text(document.page_content)

            for span in spans:

                if not span.text.strip():
                    continue

                metadata = document.metadata.copy()

                metadata["chunk_id"] = (
                    f"{metadata.get('filename', 'document')}_"
                    f"{len(chunks)}"
                )
                metadata["start_char"] = span.start
                metadata["end_char"] = span.end

                chunks.append(
                    Document(page_content=span.text.strip(), metadata=metadata)
                )

        return chunks

    def _split_text(self, text: str) -> list[_TextSpan]:

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        core_chunks: list[_TextSpan] = []
        current_chunk = ""
        current_start = None
        current_end = None
        cursor = 0

        for paragraph in paragraphs:

            para_start = text.find(paragraph, cursor)
            para_end = para_start + len(paragraph)
            cursor = para_end

            if self._fits(paragraph):

                candidate = (
                    f"{current_chunk}\n\n{paragraph}" if current_chunk else paragraph
                )

                if self._fits(candidate):
                    current_chunk = candidate
                    current_start = current_start if current_start is not None else para_start
                    current_end = para_end
                else:
                    if current_chunk:
                        core_chunks.append(_TextSpan(current_chunk, current_start, current_end))
                    current_chunk = paragraph
                    current_start = para_start
                    current_end = para_end

            else:
                if current_chunk:
                    core_chunks.append(_TextSpan(current_chunk, current_start, current_end))
                    current_chunk = ""
                    current_start = None
                    current_end = None

                core_chunks.extend(
                    self._split_large_paragraph(paragraph, base_offset=para_start)
                )

        if current_chunk:
            core_chunks.append(_TextSpan(current_chunk, current_start, current_end))

        return self._apply_overlap(core_chunks)

    def _split_large_paragraph(self, text: str, base_offset: int) -> list[_TextSpan]:

        sentences = [
            sentence.strip()
            for sentence in self.sentence_splitter(text)
            if sentence.strip()
        ]

        chunks: list[_TextSpan] = []
        current_chunk = ""
        current_start = None
        current_end = None
        cursor = 0

        for sentence in sentences:

            sent_start = text.find(sentence, cursor)
            sent_end = sent_start + len(sentence)
            cursor = sent_end

            abs_start = base_offset + sent_start
            abs_end = base_offset + sent_end

            candidate = f"{current_chunk} {sentence}" if current_chunk else sentence

            if self._fits(candidate):
                current_chunk = candidate
                current_start = current_start if current_start is not None else abs_start
                current_end = abs_end
            else:
                if current_chunk:
                    chunks.append(_TextSpan(current_chunk, current_start, current_end))

                if not self._fits(sentence):
                    chunks.extend(self._split_by_words(sentence, base_offset=abs_start))
                    current_chunk = ""
                    current_start = None
                    current_end = None
                else:
                    current_chunk = sentence
                    current_start = abs_start
                    current_end = abs_end

        if current_chunk:
            chunks.append(_TextSpan(current_chunk, current_start, current_end))

        return chunks

    def _split_by_words(self, text: str, base_offset: int) -> list[_TextSpan]:

        words = text.split()

        chunks: list[_TextSpan] = []
        current_chunk = ""
        current_start = None
        current_end = None
        cursor = 0

        for word in words:

            word_start = text.find(word, cursor)
            word_end = word_start + len(word)
            cursor = word_end

            abs_start = base_offset + word_start
            abs_end = base_offset + word_end

            candidate = f"{current_chunk} {word}" if current_chunk else word

            if self._fits(candidate):
                current_chunk = candidate
                current_start = current_start if current_start is not None else abs_start
                current_end = abs_end
            else:
                if current_chunk:
                    chunks.append(_TextSpan(current_chunk, current_start, current_end))
                current_chunk = word
                current_start = abs_start
                current_end = abs_end

        if current_chunk:
            chunks.append(_TextSpan(current_chunk, current_start, current_end))

        return chunks

    def _apply_overlap(self, spans: list[_TextSpan]) -> list[_TextSpan]:

        if self.chunk_overlap <= 0 or not spans:
            return spans

        result = [spans[0]]

        for i in range(1, len(spans)):

            overlap_text = self._tail_overlap(spans[i - 1].text)

            combined_text = (
                f"{overlap_text} {spans[i].text}" if overlap_text else spans[i].text
            )

            if overlap_text and not self._fits(combined_text):
                logger.debug(
                    "Overlap dropped for chunk [%d:%d]: combined size %d tokens "
                    "exceeds chunk_size=%d (overlap alone was %d tokens)",
                    spans[i].start,
                    spans[i].end,
                    self.token_counter(combined_text),
                    self.chunk_size,
                    self.token_counter(overlap_text),
                )
                combined_text = spans[i].text

            result.append(_TextSpan(combined_text, spans[i].start, spans[i].end))

        return result

    
    def _tail_overlap(self, text: str) -> str:
        words = text.split()

        overlap_words: list[str] = []
        for word in reversed(words):
            candidate = " ".join([word] + overlap_words)
            if self.token_counter(candidate) > self.chunk_overlap:
                break
            overlap_words.insert(0, word)

        return " ".join(overlap_words)