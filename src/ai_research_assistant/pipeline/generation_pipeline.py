from ai_research_assistant.generation.context_builder import ContextBuilder
from ai_research_assistant.generation.prompt_builder import PromptBuilder
from ai_research_assistant.generation.generator_factory import GeneratorFactory
from ai_research_assistant.citation.citation_matcher import CitationMatcher
from ai_research_assistant.citation.citation_validator import CitationValidator
from ai_research_assistant.citation.citation_formatter import CitationFormatter


class GenerationPipeline:

    def __init__(self, embedder):

        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder()
        self.generator = GeneratorFactory.create_generator()

        self.citation_matcher = CitationMatcher(embedder=embedder)
        self.citation_validator = CitationValidator()
        self.citation_formatter = CitationFormatter()

    def run(self, query: str, documents) -> dict:

        context, source_map = self.context_builder.build(documents)

        messages = self.prompt_builder.build(query=query, context=context)

        # Capture the model's true, untouched output
        raw_answer = self.generator.generate(messages)

        citation_matches = self.citation_matcher.match(
            answer=raw_answer,
            documents=documents,
            source_map=source_map
        )

        formatted_answer = self.citation_formatter.format_answer(
            answer=raw_answer,
            citation_matches=citation_matches,
            source_map=source_map
        )

        # Validate the formatted (production) answer
        citation_validation = self.citation_validator.validate(
            answer=formatted_answer,
            source_map=source_map
        )

        # Validate the raw, unmodified LLM output — the real compliance signal
        raw_citation_validation = self.citation_validator.validate(
            answer=raw_answer,
            source_map=source_map
        )

        return {
            "answer": formatted_answer,
            "raw_answer": raw_answer,
            "sources": source_map,
            "citation_matches": citation_matches,
            "citation_validation": citation_validation,
            "raw_citation_validation": raw_citation_validation
        }