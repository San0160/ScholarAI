import logging

from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.pipeline.generation_pipeline import GenerationPipeline
from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Top-level entry point: a user's query in, an answer (+ citations, + sources) out.

    Wires RetrievalPipeline -> GenerationPipeline. This is what an API layer
    or CLI should call -- neither of the two sub-pipelines is meant to be
    used standalone in production.
    """

    def __init__(self, storage_path: str = None):

        config_manager = ConfigurationManager()
        embedder = EmbeddingFactory.create_embedding(config_manager.get_embedding_config())
        self.retrieval_pipeline = RetrievalPipeline(storage_path=storage_path)
        self.generation_pipeline = GenerationPipeline(embedder=embedder)

    def run(self, query: str, metadata_filters: dict = None, min_score: float = None) -> dict:

        results = self.retrieval_pipeline.run(query, metadata_filters=metadata_filters, min_score=min_score)
        documents = [result.document for result in results]
        response = self.generation_pipeline.run(query, documents)
        logger.info("QueryPipeline.run(): answered using %d retrieved document(s)", len(documents))

        return response