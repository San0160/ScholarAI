from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.context.context_builder import ContextBuilder
from ai_research_assistant.prompts.prompt_builder import PromptBuilder
from ai_research_assistant.llm.llm_factory import LLMFactory


class QueryPipeline:

    def __init__(self):

        self.retrieval_pipeline = RetrievalPipeline()

        self.context_builder = ContextBuilder()

        self.prompt_builder = PromptBuilder()

        self.llm = LLMFactory.create_llm()

    def run(self, query: str) -> str:

        # 1. Retrieve relevant chunks
        results = self.retrieval_pipeline.run(query)

        # 2. Build context
        context = self.context_builder.build(results)

        # 3. Build prompt
        prompt = self.prompt_builder.build(
            query=query,
            context=context
        )

        # 4. Generate answer
        answer = self.llm.generate(prompt)

        return answer