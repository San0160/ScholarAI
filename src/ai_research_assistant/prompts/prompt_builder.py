from ai_research_assistant.entity.retrieval_result import RetrievalResult


class PromptBuilder:

    def build(
        self,
        query: str,
        context: str
    ) -> str:

        prompt = f"""
You are ScholarAI, an AI research assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context,
say that the information is not available in the provided documents.

Do not make up information.

Context:
--------------------
{context}
--------------------

Question:
{query}

Answer:
"""

        return prompt.strip()