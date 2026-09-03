import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from ai_research_assistant.api.schemas import IndexResponse
from ai_research_assistant.api.dependencies import get_indexing_pipeline
from ai_research_assistant.pipeline.indexing_pipeline import IndexingPipeline

router = APIRouter()

UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

@router.post("", response_model=IndexResponse)
def index_document(
    file: UploadFile = File(...),
    indexing_pipeline: IndexingPipeline = Depends(get_indexing_pipeline),
):

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    destination = UPLOAD_DIR / file.filename

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = indexing_pipeline.run_single(str(destination))

    return IndexResponse(
        filename=file.filename,
        documents=result["documents"],
        chunks=result["chunks"]
    )