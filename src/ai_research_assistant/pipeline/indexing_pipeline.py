from pathlib import Path

from ai_research_assistant.chunking.recursive_chunker import RecursiveChunker
from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.vector_store.vector_store_factory import VectorStoreFactory
from ai_research_assistant.ingestion.document_loader import DocumentLoader
from ai_research_assistant.utils.text_cleaner import TextCleaner


class IndexingPipeline:

    def __init__(self):

        config = ConfigurationManager().config

        self.loader = DocumentLoader()

        self.cleaner = TextCleaner()
        
        self.chunker = RecursiveChunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap
        )

        self.embedder = EmbeddingFactory.create_embedding()

        self.vector_store = VectorStoreFactory.create_vector_store(
            dimension=self.embedder.dimension
        )

    def run(self, file_path: str):

        # 1. Load document
        documents = self.loader.load(file_path)

        # 2. Clean text
        documents = self.cleaner.clean_documents(
            documents
        )

        # 3. Create chunks
        chunks = self.chunker.split_documents(
            documents
        )

        # 4. Generate embeddings
        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = self.embedder.embed_documents(
            texts
        )

        # 5. Add to vector store
        self.vector_store.add_documents(
            chunks,
            embeddings
        )

        # 6. Persist index
        self.vector_store.save()

        return {
            "documents": len(documents),
            "chunks": len(chunks)
        }