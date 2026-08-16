class PromptBuilder:

    SYSTEM_PROMPT = (
        "You are ScholarAI, a research assistant. "
        "Answer the user's question using only the provided context. "
        "If the context does not contain enough information to answer "
        "the question, say that the information is not available "
        "in the provided document. "
        "Do not invent facts or use information outside the "
        "provided context."
    )

    def build(
        self,
        query: str,
        context: str
    ) -> list[dict]:

        return [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{query}"
                )
            }
        ]