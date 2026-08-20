import re


class CitationFormatter:

    SOURCE_PATTERN = re.compile(r"\[Source\s+\d+\]")

    def format_source(self, source_id: int, source_map: dict) -> str:

        source = source_map.get(source_id)

        if source is None:
            return ""

        filename = source.get(
            "filename",
            "unknown"
        )

        page = source.get(
            "page",
            "unknown"
        )

        return (
            f"[Source {source_id}: "
            f"{filename}, p. {page}]"
        )

    def format_answer(
        self,
        answer: str,
        citation_matches: list[dict],
        source_map: dict
    ) -> str:

        if not answer.strip():
            return answer
        
        cleaned_answer = self.SOURCE_PATTERN.sub(
            "",
            answer
        )

        cleaned_answer = re.sub(
            r"\s+([.,!?;:])",
            r"\1",
            cleaned_answer
        )

        cleaned_answer = re.sub(
            r"\s{2,}",
            " ",
            cleaned_answer
        )

        cleaned_answer = cleaned_answer.strip()

        if not citation_matches:
            return cleaned_answer

        citations = []

        for match in citation_matches:

            source_id = match.get(
                "source_id"
            )

            citation = self.format_source(
                source_id=source_id,
                source_map=source_map
            )

            if citation:
                citations.append(
                    citation
                )

        if not citations:
            return cleaned_answer

        citation_text = " ".join(
            citations
        )

        return (
            f"{cleaned_answer} "
            f"{citation_text}"
        )