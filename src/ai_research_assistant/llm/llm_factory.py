# llm_factory.py
from functools import lru_cache

from ai_research_assistant.entity.config_entity import LLMConfig
from ai_research_assistant.llm.base_llm import BaseLLM
from ai_research_assistant.llm.huggingface_llm import HuggingFaceLLM


class LLMFactory:

    @staticmethod
    @lru_cache(maxsize=1)
    def create_llm(config: LLMConfig) -> BaseLLM:

        if config.provider == "huggingface":
            return HuggingFaceLLM(
                config.model,
                device=config.device,
                max_context_tokens=config.max_context_tokens,
                max_new_tokens=config.max_new_tokens,
            )

        raise ValueError(f"Unsupported LLM provider: {config.provider}")