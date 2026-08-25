from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.answers.answer_generator import AnswerGenerator
from ai_research_assistant.generation.context_builder import ContextBuilder


class QueryPipeline:

    def __init__(self):

        self.retrieval_pipeline = RetrievalPipeline()

        self.context_builder = ContextBuilder()

        self.answer_generator = AnswerGenerator()

    def run(
        self,
        query: str
    ) -> str:

        results = self.retrieval_pipeline.run(
            query
        )

        context = self.context_builder.build(
            results
        )

        answer = self.answer_generator.generate(
            query=query,
            context=context
        )

        return answer