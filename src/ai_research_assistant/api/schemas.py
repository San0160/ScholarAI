from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")


class CitationResponse(BaseModel):
    source: str
    page: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


class IndexResponse(BaseModel):
    filename: str
    documents: int
    chunks: int