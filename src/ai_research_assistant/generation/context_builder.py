from ai_research_assistant.entity.document import Document


class ContextBuilder:

    def build(self, documents: list[Document]) -> tuple[str, dict]:

        context_parts = []
        source_map = {}

        for source_id, document in enumerate(documents, start=1):

            filename = document.metadata.get(
                "filename",
                "unknown"
            )

            page = document.metadata.get(
                "page",
                "unknown"
            )

            context_parts.append(
                f"[Source {source_id}]\n"
                f"Document: {filename}\n"
                f"Page: {page}\n\n"
                f"{document.page_content}"
            )

            source_map[source_id] = {
                "chunk_id": document.metadata.get("chunk_id"),
                "filename": filename,
                "page": page
            }

        context = "\n\n".join(context_parts)

        return context, source_map