import logging
import uuid

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ai_research_assistant.constants import RAW_DATA_PATH
from ai_research_assistant.api.schemas import IndexResponse
from ai_research_assistant.api.dependencies import get_indexing_pipeline
from ai_research_assistant.pipeline.indexing_pipeline import IndexingPipeline

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = RAW_DATA_PATH  
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
READ_CHUNK_BYTES = 1024 * 1024


@router.post("", response_model=IndexResponse)
def index_document(
    file: UploadFile = File(...),
    rebuild: bool = Form(False),
    indexing_pipeline: IndexingPipeline = Depends(get_indexing_pipeline),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    safe_filename = Path(file.filename).name  # strips any directory components

    extension = Path(safe_filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    destination = UPLOAD_DIR / safe_filename

    if destination.exists() and not rebuild:
        raise HTTPException(
            status_code=409,
            detail=f"A file named '{safe_filename}' already exists. "
                   f"Pass rebuild=true to re-index it, or rename and try again."
        )

    # Write to a temp file first so a failed/oversized upload never leaves a
    # truncated file at `destination` -- important once rebuild=True can overwrite
    # a previously-working file.
    tmp_path = UPLOAD_DIR / f".{uuid.uuid4().hex}.part"

    total_bytes = 0
    try:
        with tmp_path.open("wb") as buffer:
            while chunk := file.file.read(READ_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB upload limit."
                    )
                buffer.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        tmp_path.replace(destination)  # atomic swap -- only now touches the real path
    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise

    try:
        result = indexing_pipeline.run_single(str(destination), rebuild=rebuild)
    except Exception:
        destination.unlink(missing_ok=True)  # don't leave an unindexed file blocking retries
        raise

    return IndexResponse(
        filename=safe_filename,
        documents=result["documents"],
        chunks=result["chunks"]
    )