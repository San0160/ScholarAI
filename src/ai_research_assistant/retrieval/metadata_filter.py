from ai_research_assistant.entity.retrieval_result import RetrievalResult


class MetadataFilter:

    def filter(
        self,
        results: list[RetrievalResult]
    ) -> list[RetrievalResult]:

        return results