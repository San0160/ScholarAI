from fastapi import APIRouter, Depends, HTTPException

from ai_research_assistant.retrieval.query_intent import is_whole_document_question
from ai_research_assistant.api.schemas import QueryRequest, QueryResponse, CitationResponse
from ai_research_assistant.api.dependencies import get_retrieval_pipeline, get_generation_pipeline
from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.pipeline.generation_pipeline import GenerationPipeline

router = APIRouter()


@router.post("", response_model=QueryResponse)
def query(
    request: QueryRequest,
    retrieval_pipeline: RetrievalPipeline = Depends(get_retrieval_pipeline),
    generation_pipeline: GenerationPipeline = Depends(get_generation_pipeline),
):

    retrieved_results = retrieval_pipeline.run(
        request.question,
        metadata_filters=request.metadata_filters,
        top_k=request.top_k,
    )

    if not retrieved_results:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    documents = []

    # Summary-style question about one file: read across the file instead of searching it.
    if request.metadata_filters and is_whole_document_question(request.question):
        chunks = retrieval_pipeline.get_document_chunks(request.metadata_filters)
        documents = generation_pipeline.select_overview_documents(chunks)

    if not documents:
        retrieved_results = retrieval_pipeline.run(
            request.question,
            metadata_filters=request.metadata_filters,
            top_k=request.top_k,
        )
        if not retrieved_results:
            raise HTTPException(status_code=404, detail="No relevant documents found.")
        documents = [r.document for r in retrieved_results]

    result = generation_pipeline.run(query=request.question, documents=documents)

    # Show the sources the model actually cited; fall back to everything it was given.
    sources = result["sources"]
    validation = result.get("raw_citation_validation") or {}
    cited_ids = [i for i in validation.get("cited_sources", []) if i in sources]
    shown = [sources[i] for i in cited_ids] or list(sources.values())

    citations = [
        CitationResponse(
            source=source.get("filename", "unknown"),
            page=source.get("page") if isinstance(source.get("page"), int) else None,
            start_char=source.get("start_char"),
            end_char=source.get("end_char"),
        )
        for source in shown
    ]

    return QueryResponse(answer=result["answer"], citations=citations)