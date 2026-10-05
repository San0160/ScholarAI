import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class DocumentNotFoundError(Exception):
    def __init__(self, file_path: str):
        super().__init__(f"Document not found: {file_path}")
        self.file_path = file_path


class RetrievalError(Exception):
    pass


class GenerationError(Exception):
    pass


class IndexingError(Exception):
    pass


async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
    logger.warning("DocumentNotFoundError on %s %s: %s", request.method, request.url.path, exc.file_path)
    return JSONResponse(
        status_code=404,
        content={"detail": f"Document not found: {exc.file_path}"}
    )


async def retrieval_error_handler(request: Request, exc: RetrievalError):
    logger.error("RetrievalError on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Retrieval failed. Please try again."}
    )


async def generation_error_handler(request: Request, exc: GenerationError):
    logger.error("GenerationError on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=502,
        content={"detail": "Answer generation failed. Please try again."}
    )


async def indexing_error_handler(request: Request, exc: IndexingError):
    logger.error("IndexingError on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Indexing failed. Please try again."}
    )