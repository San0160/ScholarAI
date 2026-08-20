from ai_research_assistant.citation.citation import Citation


class SourceMapper:

    def build(
        self,
        documents
    ) -> dict[int, Citation]:

        sources = {}

        for source_id, document in enumerate(
            documents,
            start=1
        ):

            metadata = document.metadata

            citation = Citation(
                source_id=source_id,
                document=metadata.get(
                    "filename",
                    "unknown"
                ),
                page=metadata.get(
                    "page",
                    "unknown"
                ),
                chunk_id=metadata.get(
                    "chunk_id",
                    "unknown"
                )
            )

            sources[source_id] = citation

        return sources