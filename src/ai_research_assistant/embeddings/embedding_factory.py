from functools import lru_cache

from ai_research_assistant.embeddings.api_embedding import APIEmbedding
from ai_research_assistant.embeddings.base_embedding import BaseEmbedding
from ai_research_assistant.embeddings.huggingface_embedding import HuggingFaceEmbedding
from ai_research_assistant.entity.config_entity import EmbeddingConfig


class EmbeddingFactory:
    @staticmethod
    @lru_cache(maxsize=1)
    def create_embedding(config: EmbeddingConfig) -> BaseEmbedding:
        if config.provider == "huggingface":
            return HuggingFaceEmbedding(
                config.model, device=config.device,
                batch_size=config.batch_size, show_progress_bar=config.show_progress_bar,
            )
        if config.provider == "api":
            return APIEmbedding(
                model=config.model, base_url=config.api_base_url,
                api_key_env_var=config.api_key_env_var, dimension=config.dimension,
            )
        raise ValueError(f"Unsupported embedding provider: {config.provider}")