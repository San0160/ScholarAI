import logging

from ai_research_assistant.entity.retrieval_result import RetrievalResult

logger = logging.getLogger(__name__)


class MetadataFilter:

    def filter(
        self,
        results: list[RetrievalResult],
        metadata_filters: dict | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:

        filtered = results

        if metadata_filters:
            filtered = [
                result for result in filtered
                if all(
                    result.document.metadata.get(key) == value
                    for key, value in metadata_filters.items()
                )
            ]

        if min_score is not None:
            filtered = [
                result for result in filtered
                if result.score >= min_score
            ]

        if len(filtered) != len(results):
            logger.debug(
                "MetadataFilter: %d/%d results dropped (metadata_filters=%s, min_score=%s)",
                len(results) - len(filtered), len(results), metadata_filters, min_score,
            )

        return filtered