from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.vector_store.vector_store_factory import VectorStoreFactory
from ai_research_assistant.retrieval.vector_retriver import VectorRetriever


class RetrievalPipeline:

    def __init__(self):

        config = ConfigurationManager().config

        self.embedder = EmbeddingFactory.create_embedding()

        self.vector_store = VectorStoreFactory.create_vector_store(
            dimension=self.embedder.dimension
        )

        self.retriever = VectorRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=config.retrieval.top_k,
            similarity_threshold=config.retrieval.similarity_threshold
        )

    def retrieve(self, query: str):

        return self.retriever.retrieve(query)