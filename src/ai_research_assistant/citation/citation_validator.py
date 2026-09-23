import logging

from ai_research_assistant.utils.citation_utils import SOURCE_BLOCK_PATTERN, extract_source_ids

logger = logging.getLogger(__name__)


class CitationValidator:

    def validate(self, answer: str, source_map: dict) -> dict:
        cited_numbers = set()
        for block in SOURCE_BLOCK_PATTERN.findall(answer):
            cited_numbers.update(extract_source_ids(block))

        valid_numbers = set(source_map.keys())
        invalid_citations = cited_numbers - valid_numbers

        return {
            "cited_sources": sorted(cited_numbers),
            "invalid_citations": sorted(invalid_citations),
            "has_citations": len(cited_numbers) > 0,
            "all_citations_valid": len(invalid_citations) == 0,
            "citation_count": len(cited_numbers),
        }