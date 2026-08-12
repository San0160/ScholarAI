from ai_research_assistant.entity.retrieval_result import RetrievalResult


class MetadataFilter:

    def filter(
        self,
        results: list[RetrievalResult]
    ) -> list[RetrievalResult]:

        filtered_results = []

        for result in results:

            section = result.document.metadata.get(
                "section",
                ""
            ).lower()

            if section == "references":
                continue

            filtered_results.append(result)

        return filtered_results