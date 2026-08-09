from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.huggingface_embedding import HuggingFaceEmbedding


class EmbeddingFactory:

    @staticmethod
    def create_embedding():

        config = ConfigurationManager().config

        provider = config.embeddings.provider
        model_name = config.embeddings.model

        if provider == "huggingface":
            return HuggingFaceEmbedding(model_name)

        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )