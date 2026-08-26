from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException
import os

from ai_research_assistant.api.schemas import IndexRequest, IndexResponse
from ai_research_assistant.api.dependencies import get_indexing_pipeline
from ai_research_assistant.pipeline.indexing_pipeline import IndexingPipeline

router = APIRouter()


@router.post("", response_model=IndexResponse)
def index_document(
    request: IndexRequest,
    indexing_pipeline: IndexingPipeline = Depends(get_indexing_pipeline),
):

    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")

    result = indexing_pipeline.run(request.file_path)

    return IndexResponse(
        documents=result["documents"],
        chunks=result["chunks"]
    )