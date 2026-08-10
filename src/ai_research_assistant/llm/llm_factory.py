from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.llm.huggingface_llm import HuggingFaceLLM


class LLMFactory:

    @staticmethod
    def create_llm():

        config = ConfigurationManager().config

        provider = config.llm.provider
        model_name = config.llm.model

        if provider == "huggingface":
            return HuggingFaceLLM(model_name)

        raise ValueError(
            f"Unsupported LLM provider: {provider}"
        )