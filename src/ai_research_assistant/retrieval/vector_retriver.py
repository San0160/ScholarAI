import logging
import time

from ai_research_assistant.entity.retrieval_result import RetrievalResult
from ai_research_assistant.retrieval.base_retriver import BaseRetriever

logger = logging.getLogger(__name__)

class VectorRetriever(BaseRetriever):

    def __init__(
        self,
        embedder,
        vector_store,
        top_k: int
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> list[RetrievalResult]:

        if not query or not query.strip():
            raise ValueError("query must not be empty")

        effective_top_k = top_k if top_k is not None else self.top_k

        start = time.perf_counter()

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.similarity_search(
            query_embedding,
            effective_top_k
        )

        elapsed = time.perf_counter() - start

        logger.info(
            "Retrieved %d result(s) for query (%d chars) in %.3fs (top_k=%d)",
            len(results),
            len(query),
            elapsed,
            effective_top_k,
        )

        return results