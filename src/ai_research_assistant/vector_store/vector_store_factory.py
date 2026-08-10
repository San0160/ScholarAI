from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.vector_store.faiss_vector_store import FAISSVectorStore


class VectorStoreFactory:

    @staticmethod
    def create_vector_store(dimension: int):

        config = ConfigurationManager().config

        provider = config.vector_store.provider
        storage_path = config.vector_store.path

        if provider == "faiss":
            return FAISSVectorStore(
                dimension=dimension,
                storage_path=storage_path
            )

        raise ValueError(
            f"Unsupported vector store provider: {provider}"
        )