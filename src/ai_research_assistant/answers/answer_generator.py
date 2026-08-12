from ai_research_assistant.llm.llm_factory import LLMFactory
from ai_research_assistant.prompts.prompt_builder import PromptBuilder


class AnswerGenerator:

    def __init__(self):

        self.llm = LLMFactory.create_llm()
        self.prompt_builder = PromptBuilder()

    def generate(
        self,
        query: str,
        context: str
    ) -> str:

        prompt = self.prompt_builder.build(
            query=query,
            context=context
        )

        return self.llm.generate(prompt)