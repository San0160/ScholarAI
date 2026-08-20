from ai_research_assistant.generation.context_builder import (
    ContextBuilder
)

from ai_research_assistant.prompts.prompt_builder import (
    PromptBuilder
)

from ai_research_assistant.generation.llm_generator import (
    LLMGenerator
)

from ai_research_assistant.citation.citation_matcher import (
    CitationMatcher
)

from ai_research_assistant.citation.citation_validator import (
    CitationValidator
)

from ai_research_assistant.citation.citation_formatter import (
    CitationFormatter
)


class GenerationPipeline:

    def __init__(
        self,
        model_name: str,
        embedder
    ):

        self.context_builder = ContextBuilder()

        self.prompt_builder = PromptBuilder()

        self.generator = LLMGenerator(
            model_name=model_name
        )

        self.citation_matcher = CitationMatcher(
            embedder=embedder
        )

        self.citation_validator = CitationValidator()

        self.citation_formatter = CitationFormatter()


    def run(
        self,
        query: str,
        documents
    ) -> dict:

        # ------------------------------------------
        # Build Context + Source Map
        # ------------------------------------------

        context, source_map = (
            self.context_builder.build(
                documents
            )
        )


        # ------------------------------------------
        # Build Prompt
        # ------------------------------------------

        messages = self.prompt_builder.build(
            query=query,
            context=context
        )


        # ------------------------------------------
        # Generate Answer
        # ------------------------------------------

        answer = self.generator.generate(
            messages
        )


        # ------------------------------------------
        # Citation Matching
        # ------------------------------------------

        citation_matches = (
            self.citation_matcher.match(
                answer=answer,
                documents=documents,
                source_map=source_map
            )
        )

        # --------------------------------------------
        # Citation Formatter
        # --------------------------------------------

        formatted_answer = self.citation_formatter.format_answer(
            answer=answer,
            citation_matches=citation_matches,
            source_map=source_map
        )


        # ------------------------------------------
        # Citation Validation
        # ------------------------------------------

        citation_validation = (
            self.citation_validator.validate(
                answer= formatted_answer,
                source_map=source_map
            )
        )


        # ------------------------------------------
        # Return Generation Result
        # ------------------------------------------

        return {
            "answer": formatted_answer,
            "sources": source_map,
            "citation_matches": citation_matches,
            "citation_validation": citation_validation
        }