# generator_factory.py
from ai_research_assistant.entity.config_entity import LLMConfig
from ai_research_assistant.generation.base_generator import BaseGenerator
from ai_research_assistant.generation.llm_generator import LLMGenerator
from ai_research_assistant.llm.llm_factory import LLMFactory


class GeneratorFactory:

    @staticmethod
    def create_generator(llm_config: LLMConfig) -> BaseGenerator:

        llm = LLMFactory.create_llm(llm_config)

        return LLMGenerator(llm=llm)