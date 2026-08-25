from ai_research_assistant.generation.base_generator import BaseGenerator
from ai_research_assistant.llm.base_llm import BaseLLM


class LLMGenerator(BaseGenerator):

    def __init__(
        self,
        llm: BaseLLM
    ):

        self.llm = llm

    def generate(
        self,
        messages: list[dict]
    ) -> str:

        if not messages:

            return (
                "The information is not available "
                "in the provided document."
            )

        return self.llm.generate(messages)