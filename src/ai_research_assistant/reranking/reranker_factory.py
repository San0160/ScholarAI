# reranking/reranker_factory.py
from functools import lru_cache

from ai_research_assistant.entity.config_entity import RerankingConfig
from ai_research_assistant.reranking.api_reranker import APIReranker
from ai_research_assistant.reranking.base_reranker import BaseReranker
from ai_research_assistant.reranking.bge_reranker import BGEReranker
from ai_research_assistant.reranking.cross_encoder_reranker import CrossEncoderReranker


class RerankerFactory:

    @staticmethod
    @lru_cache(maxsize=1)
    def create_reranker(config: RerankingConfig) -> BaseReranker:

        if config.provider == "bge":
            return BGEReranker(config.model)

        if config.provider == "cross_encoder":
            return CrossEncoderReranker(config.model)

        if config.provider == "api":
            return APIReranker(
                model=config.model,
                base_url=config.api_base_url,
                api_key_env_var=config.api_key_env_var,
            )

        raise ValueError(f"Unsupported reranking provider: {config.provider}")