from dataclasses import dataclass

from ai_research_assistant.retrieval.query_intent import is_whole_document_question


@dataclass
class AnswerOutcome:
    """Everything produced when one question is answered."""

    answer: str                  # the text shown to the user
    citations: list[dict]        # sources shown to the user (filename, page, start_char, end_char)
    model_cited: bool            # True if the model itself cited at least one source
    context: str                 # the excerpts that were put in the prompt
    context_sources: list[dict]  # every source that made it into the prompt
    generation: dict             # the full generation pipeline result


def answer_question(
    retrieval_pipeline,
    generation_pipeline,
    question: str,
    metadata_filters: dict | None = None,
    top_k: int | None = None,
) -> AnswerOutcome | None:
    """Answer one question. Used by the API and by the evaluation scripts, so
    both run exactly the same steps.

    Returns None when there is nothing to answer from (no matching documents).
    """
    documents = []

    # Summary-style question about one file: read across the file instead of searching it.
    if metadata_filters and is_whole_document_question(question):
        chunks = retrieval_pipeline.get_document_chunks(metadata_filters)
        documents = generation_pipeline.select_overview_documents(chunks)

    if not documents:
        results = retrieval_pipeline.run(
            question,
            metadata_filters=metadata_filters,
            top_k=top_k,
        )
        documents = [result.document for result in results]

    if not documents:
        return None

    result = generation_pipeline.run(query=question, documents=documents)

    # Show the sources the model actually cited; fall back to everything it was given.
    sources = result["sources"]
    validation = result.get("raw_citation_validation") or {}
    cited_ids = [i for i in validation.get("cited_sources", []) if i in sources]
    shown = [sources[i] for i in cited_ids] or list(sources.values())

    return AnswerOutcome(
        answer=result["answer"],
        citations=shown,
        model_cited=bool(cited_ids),
        context=result.get("context", ""),
        context_sources=list(sources.values()),
        generation=result,
    )
