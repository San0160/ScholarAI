from ai_research_assistant.entity.document import Document
from ai_research_assistant.chunking.base_chunker import BaseChunker


class RecursiveChunker(BaseChunker):

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(
        self,
        documents: list[Document]
    ) -> list[Document]:

        chunks = []

        for document in documents:

            text = document.page_content

            start = 0

            while start < len(text):

                end = start + self.chunk_size

                chunk = text[start:end]

                metadata = document.metadata.copy()

                metadata["chunk_start"] = start
                metadata["chunk_end"] = end
                metadata["chunk_id"] = f"{metadata['filename']}_{len(chunks)}"

                chunks.append(
                    Document(
                        page_content=chunk,
                        metadata=metadata
                    )
                )

                start += self.chunk_size - self.chunk_overlap

        return chunks