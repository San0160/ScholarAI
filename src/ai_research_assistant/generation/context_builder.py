from ai_research_assistant.entity.document import Document


class ContextBuilder:

    def build(
        self,
        documents: list[Document]
    ) -> str:

        context_parts = []

        for index, document in enumerate(
            documents,
            start=1
        ):

            filename = document.metadata.get(
                "filename",
                "unknown"
            )

            page = document.metadata.get(
                "page",
                "unknown"
            )

            context_parts.append(
                f"[Source {index}]\n"
                f"Document: {filename}\n"
                f"Page: {page}\n\n"
                f"{document.page_content}"
            )

        return "\n\n".join(
            context_parts
        )