import logging

from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.entity.retrieval_result import RetrievalResult
from ai_research_assistant.retrieval.metadata_filter import MetadataFilter
from ai_research_assistant.retrieval.vector_retriver import VectorRetriever
from ai_research_assistant.vector_store.vector_store_factory import VectorStoreFactory

# from ai_research_assistant.reranking.cross_encoder_reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Embeds a query, retrieves candidate chunks from the vector store,
    and applies metadata filtering. Reranking is not yet wired in (see
    the commented-out CrossEncoderReranker below), so final_top_k
    currently truncates by raw cosine similarity, not a reranker score.
    """

    def __init__(self, storage_path: str = None):

        config_manager = ConfigurationManager()
        config = config_manager.config  # raw access for sections without an entity yet

        self.embedder = EmbeddingFactory.create_embedding(
            config_manager.get_embedding_config()
        )

        self.vector_store = VectorStoreFactory.create_vector_store(
            dimension=self.embedder.dimension,
            storage_path=storage_path,
        )

        try:
            self.vector_store.load()
        except FileNotFoundError as error:
            raise RuntimeError(
                "No vector store found -- run the indexing pipeline before "
                "starting retrieval"
            ) from error

        self.retriever = VectorRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=config.retrieval.candidate_k
        )

        self.metadata_filter = MetadataFilter()

        # Reranker disabled for FAISS baseline
        # self.reranker = CrossEncoderReranker(
        #     model_name=config.reranking.model
        # )

        self.final_top_k = config.reranking.top_k

    def run(
        self,
        query: str,
        metadata_filters: dict | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:

        candidates = self.retriever.retrieve(query)
        retrieved_count = len(candidates)

        candidates = self.metadata_filter.filter(
            candidates,
            metadata_filters=metadata_filters,
            min_score=min_score,
        )

        # Reranker disabled -- truncating by raw similarity score for now
        results = candidates[: self.final_top_k]

        logger.info(
            "run(): retrieved %d -> %d after filtering -> %d returned",
            retrieved_count, len(candidates), len(results),
        )

        return results