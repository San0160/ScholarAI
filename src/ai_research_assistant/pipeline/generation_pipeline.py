from ai_research_assistant.generation.context_builder import ContextBuilder
from ai_research_assistant.generation.prompt_builder import PromptBuilder
from ai_research_assistant.generation.llm_generator import LLMGenerator


class GenerationPipeline:

    def __init__(
        self,
        model_name: str
    ):

        self.context_builder = ContextBuilder()

        self.prompt_builder = PromptBuilder()

        self.generator = LLMGenerator(
            model_name=model_name
        )

    def run(
        self,
        query: str,
        documents
    ) -> str:

        context = self.context_builder.build(
            documents
        )

        messages = self.prompt_builder.build(
            query=query,
            context=context
        )

        answer = self.generator.generate(
            messages
        )

        return answer