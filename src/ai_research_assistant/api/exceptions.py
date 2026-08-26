from fastapi import Request
from fastapi.responses import JSONResponse


class DocumentNotFoundError(Exception):
    def __init__(self, file_path: str):
        self.file_path = file_path


class RetrievalError(Exception):
    pass


async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Document not found: {exc.file_path}"}
    )


async def retrieval_error_handler(request: Request, exc: RetrievalError):
    return JSONResponse(
        status_code=500,
        content={"detail": "Retrieval failed. Please try again."}
    )