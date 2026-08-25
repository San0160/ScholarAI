from ai_research_assistant.llm.llm_factory import LLMFactory
from ai_research_assistant.generation.llm_generator import LLMGenerator


class GeneratorFactory:

    @staticmethod
    def create_generator():

        llm = LLMFactory.create_llm()

        return LLMGenerator(llm=llm)