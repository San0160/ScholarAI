# pipeline/retrieval_pipeline.py
import logging

from ai_research_assistant.api.exceptions import RetrievalError
from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.entity.retrieval_result import RetrievalResult
from ai_research_assistant.reranking.reranker_factory import RerankerFactory
from ai_research_assistant.retrieval.metadata_filter import MetadataFilter
from ai_research_assistant.retrieval.vector_retriver import VectorRetriever
from ai_research_assistant.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class RetrievalPipeline:

    def __init__(self, storage_path: str = None):

        config_manager = ConfigurationManager()
        config = config_manager.config

        self.embedder = EmbeddingFactory.create_embedding(
            config_manager.get_embedding_config()
        )

        self.vector_store = VectorStoreFactory.create_vector_store(
            dimension=self.embedder.dimension,
            storage_path=storage_path,
        )

        try:
            self.vector_store.load()
        except FileNotFoundError:
            logger.info(
                "No existing vector store found at startup -- retrieval will return "
                "no results until the indexing pipeline has processed at least one document"
            )

        self.retriever = VectorRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=config.retrieval.candidate_k
        )

        self.metadata_filter = MetadataFilter()

        self.reranker = RerankerFactory.create_reranker(
            config_manager.get_reranking_config()
        )

        self.final_top_k = config.reranking.top_k

    def run(
        self,
        query: str,
        metadata_filters: dict | None = None,
        min_score: float | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:

        try:
            candidates = self.retriever.retrieve(query)
        except Exception as error:
            logger.exception("Retrieval failed for query: %r", query)
            raise RetrievalError("Failed to retrieve documents for this query.") from error

        retrieved_count = len(candidates)

        candidates = self.metadata_filter.filter(
            candidates,
            metadata_filters=metadata_filters,
            min_score=min_score,
        )

        effective_top_k = top_k if top_k is not None else self.final_top_k
        results = self.reranker.rerank(query, candidates, effective_top_k)

        logger.info(
            "run(): retrieved %d -> %d after filtering -> %d after reranking (top_k=%d)",
            retrieved_count, len(candidates), len(results), effective_top_k,
        )

        return results

    def get_document_chunks(self, metadata_filters: dict) -> list:
        return self.vector_store.get_documents(metadata_filters)