from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    file_path: str = Field(..., description="Path to the document to index")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")


class CitationResponse(BaseModel):
    source: str
    page: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


class IndexResponse(BaseModel):
    documents: int
    chunks: int