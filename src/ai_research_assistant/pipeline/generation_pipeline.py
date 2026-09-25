import logging

from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.generation.context_builder import ContextBuilder
from ai_research_assistant.generation.prompt_builder import PromptBuilder
from ai_research_assistant.generation.generator_factory import GeneratorFactory
from ai_research_assistant.citation.citation_matcher import CitationMatcher
from ai_research_assistant.citation.citation_validator import CitationValidator
from ai_research_assistant.citation.citation_formatter import CitationFormatter
from ai_research_assistant.entity.document import Document
from ai_research_assistant.utils.tokenizer_utils import get_token_counter


logger = logging.getLogger(__name__)

_PROMPT_SAFETY_MARGIN_TOKENS = 200  # headroom for the question + chat-template overhead
_CITATION_OK = "ok"
_CITATION_SKIPPED = "skipped"
_CITATION_FAILED = "failed"

class GenerationPipeline:

    def __init__(self, embedder):

        config_manager = ConfigurationManager()
        llm_config = config_manager.get_llm_config()

        token_counter = get_token_counter(llm_config.model)

        system_prompt_tokens = token_counter(PromptBuilder.SYSTEM_PROMPT)
        context_budget = (
            llm_config.max_context_tokens
            - system_prompt_tokens
            - _PROMPT_SAFETY_MARGIN_TOKENS
        )

        if context_budget <= 0:
            raise ValueError(
                f"max_context_tokens ({llm_config.max_context_tokens}) leaves no "
                f"room for context after the system prompt ({system_prompt_tokens} "
                f"tokens) and safety margin ({_PROMPT_SAFETY_MARGIN_TOKENS} tokens) "
                f"-- increase max_context_tokens in config"
            )

        self.context_builder = ContextBuilder(
            token_counter=token_counter,
            max_context_tokens=context_budget,
        )

        self.prompt_builder = PromptBuilder()
        self.generator = GeneratorFactory.create_generator(llm_config)

        self.citation_matcher = CitationMatcher(embedder = embedder)
        self.citation_validator = CitationValidator()
        self.citation_formatter = CitationFormatter()

    def run(self, query: str, documents) -> dict:

        if not documents:
            logger.info("No documents provided -- skipping generation")
            return self._no_context_result()

        context, source_map = self.context_builder.build(documents)

        if not context:
            logger.warning("Context is empty after budget filtering -- skipping generation")
            return self._no_context_result()

        cited_documents = documents[: len(source_map)]

        messages = self.prompt_builder.build(query=query, context=context)

        # Capture the model's true, untouched output
        raw_answer = self.generator.generate(messages)

        try:
            citation_matches = self.citation_matcher.match(
                answer=raw_answer,
                documents=cited_documents,
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

            citation_status = _CITATION_OK

        except Exception:
            logger.exception(
                "Citation processing failed after generation succeeded -- "
                "returning raw answer without citation formatting/validation"
            )
            formatted_answer = raw_answer
            citation_matches = []
            citation_validation = None
            raw_citation_validation = None
            citation_status = _CITATION_FAILED

        logger.info(
            "run(): %d document(s) -> %d char answer, citation_status=%s",
            len(documents), len(formatted_answer), citation_status,
        )

        return {
            "answer": formatted_answer,
            "raw_answer": raw_answer,
            "sources": source_map,
            "citation_matches": citation_matches,
            "citation_validation": citation_validation,
            "raw_citation_validation": raw_citation_validation,
            "citation_status": citation_status,
        }

    @staticmethod
    def _no_context_result() -> dict:
        return {
            "answer": "The information is not available in the provided document.",
            "raw_answer": None,
            "sources": {},
            "citation_matches": [],
            "citation_validation": None,
            "raw_citation_validation": None,
            "citation_status": _CITATION_SKIPPED,
        }