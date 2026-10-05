import re

from ai_research_assistant.utils.citation_utils import SOURCE_BLOCK_PATTERN


class CitationFormatter:

    def format_answer(self, answer: str, citation_matches: list[dict], source_map: dict) -> str:
        """Return the answer as shown to the user: the model's inline [Source N]
        markers removed and whitespace tidied. Citations travel separately in the
        API's citations list, so nothing is appended here. citation_matches and
        source_map are kept only so existing callers don't change."""
        if not answer.strip():
            return answer

        cleaned_answer = SOURCE_BLOCK_PATTERN.sub("", answer)
        cleaned_answer = re.sub(r"[ \t]+([.,!?;:])", r"\1", cleaned_answer)
        cleaned_answer = re.sub(r"[ \t]{2,}", " ", cleaned_answer)
        cleaned_answer = re.sub(r"[ \t]+\n", "\n", cleaned_answer)
        cleaned_answer = re.sub(r"\n{3,}", "\n\n", cleaned_answer)

        return cleaned_answer.strip()