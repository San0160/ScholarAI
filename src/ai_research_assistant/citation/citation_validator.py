import re


class CitationValidator:

    CITATION_PATTERN = re.compile(
        r"\[Source (\d+)(?::[^\]]+)?\]"
    )

    def validate(
        self,
        answer: str,
        source_map: dict
    ) -> dict:

        cited_numbers = {
            int(match)
            for match in self.CITATION_PATTERN.findall(
                answer
            )
        }

        valid_numbers = set(
            source_map.keys()
        )

        invalid_citations = (
            cited_numbers - valid_numbers
        )

        return {
            "cited_sources": sorted(
                cited_numbers
            ),
            "invalid_citations": sorted(
                invalid_citations
            ),
            "has_citations": len(
                cited_numbers
            ) > 0,
            "all_citations_valid": (
                len(invalid_citations) == 0
            ),
            "citation_count": len(
                cited_numbers
            )
        }