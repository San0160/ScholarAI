from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.vector_store.vector_store_factory import VectorStoreFactory
from ai_research_assistant.retrieval.vector_retriver import VectorRetriever
from ai_research_assistant.reranking.cross_encoder_reranker import CrossEncoderReranker


class RetrievalPipeline:

    def __init__(self):

        config = ConfigurationManager().config

        # Create embedding model
        self.embedder = EmbeddingFactory.create_embedding()

        # Create vector store
        self.vector_store = VectorStoreFactory.create_vector_store(
            dimension=self.embedder.dimension
        )

        # Load existing index
        self.vector_store.load()

        # Create retriever
        self.retriever = VectorRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=config.retrieval.candidate_k           
        )

        self.reranker = CrossEncoderReranker(
            model_name=config.reranking.model
        )

        self.final_top_k = config.reranking.top_k

    def run(self, query: str):

        candidates = self.retriever.retrieve(query)

        return self.reranker.rerank(
            query=query,
            results=candidates,
            top_k=self.final_top_k
        )