from ai_research_assistant.entity.retrieval_result import RetrievalResult


class ContextBuilder:

    def build(self, results) -> str:

        context_parts = []

        for result in results:

            document = result.document

            metadata = document.metadata

            source = metadata.get(
                "filename",
                "unknown"
            )

            page = metadata.get(
                "page",
                "unknown"
            )

            context_parts.append(
                f"[Source: {source} | Page: {page}]\n"
                f"{document.page_content}"
            )

        return "\n\n".join(context_parts)