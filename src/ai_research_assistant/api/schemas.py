from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The user's question")
    top_k: int | None = Field(
        default=None, ge=1, le=20,
        description="Number of final results to return after reranking. Defaults to the server's configured reranking.top_k."
    )
    metadata_filters: dict[str, str | int] | None = Field(
        default=None,
        description="Restrict retrieval to documents matching these metadata fields exactly, e.g. {'filename': 'roadmap.pdf'} or {'page': 3}"
    )


class CitationResponse(BaseModel):
    source: str
    page: int | None = None          # populated for PDF sources only
    start_char: int | None = None    # character offset into the source document, works for every format
    end_char: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


class IndexResponse(BaseModel):
    filename: str
    documents: int
    chunks: int