import logging
from typing import Callable

from ai_research_assistant.entity.document import Document

logger = logging.getLogger(__name__)

class ContextBuilder:

    """Assembles retrieved chunks into a single context string with
    [Source N] labels, stopping once max_context_tokens is reached.

    max_context_tokens here is the budget for the CONTEXT portion only --
    it does NOT include the system prompt, the question, or chat-template
    formatting overhead. Callers must reserve headroom for those
    separately (pass something meaningfully smaller than the LLM's full
    max_context_tokens, not the same number).
    """

    def __init__(
        self,
        token_counter: Callable[[str], int],
        max_context_tokens: int,
    ):
        self.token_counter = token_counter
        self.max_context_tokens = max_context_tokens

    def build(self, documents: list[Document]) -> tuple[str, dict]:

        context_parts = []
        source_map = {}
        used_tokens = 0
        included = 0

        for source_id, document in enumerate(documents, start=1):

            filename = document.metadata.get("filename", "unknown")
            page = document.metadata.get("page", "unknown")

            part = (
                f"[Source {source_id}]\n"
                f"Document: {filename}\n"
                f"Page: {page}\n\n"
                f"{document.page_content}"
            )

            part_tokens = self.token_counter(part)

            if used_tokens + part_tokens > self.max_context_tokens:
                logger.warning(
                    "Context budget reached after %d/%d document(s) "
                    "(%d/%d tokens) -- dropping remaining lower-ranked results",
                    included, len(documents), used_tokens, self.max_context_tokens,
                )
                break

            context_parts.append(part)
            source_map[source_id] = {
                "chunk_id": document.metadata.get("chunk_id"),
                "filename": filename,
                "page": page
            }
            
            used_tokens += part_tokens
            included += 1

        context = "\n\n".join(context_parts)

        return context, source_map