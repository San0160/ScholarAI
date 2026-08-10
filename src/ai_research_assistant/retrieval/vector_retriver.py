from ai_research_assistant.entity.retrieval_result import RetrievalResult
from ai_research_assistant.retrieval.base_retriver import BaseRetriever


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
        query: str
    ) -> list[RetrievalResult]:

        query_embedding = self.embedder.embed_query(query)

        return self.vector_store.similarity_search(
            query_embedding,
            self.top_k
        )