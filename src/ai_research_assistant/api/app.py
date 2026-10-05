from dotenv import load_dotenv
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware  

from ai_research_assistant.api.exceptions import (
    DocumentNotFoundError,
    document_not_found_handler,
    RetrievalError,
    retrieval_error_handler,
    GenerationError,
    generation_error_handler,
    IndexingError,
    indexing_error_handler,
)
from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.pipeline.generation_pipeline import GenerationPipeline
from ai_research_assistant.pipeline.indexing_pipeline import IndexingPipeline
from ai_research_assistant.api.routers import index_router, query_router
from ai_research_assistant.constants import *
from ai_research_assistant.logging.logger import logger

load_dotenv()

def _reset_data_on_startup():
    logger.warning("RESET_ON_STARTUP is set -- wiping raw uploads and vector store for a clean start")

    if RAW_DATA_PATH.exists():
        shutil.rmtree(RAW_DATA_PATH)
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

    if VECTOR_DB_PATH.exists():
        shutil.rmtree(VECTOR_DB_PATH)
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Loading models and pipelines...")

    if os.environ.get("RESET_ON_STARTUP", "false").lower() == "true":
        _reset_data_on_startup()

    config_manager = ConfigurationManager()
    embedder = EmbeddingFactory.create_embedding(config_manager.get_embedding_config())

    app.state.embedder = embedder
    app.state.retrieval_pipeline = RetrievalPipeline()
    app.state.generation_pipeline = GenerationPipeline(embedder=embedder)
    app.state.indexing_pipeline = IndexingPipeline()

    logger.info("Startup complete.")

    yield

    logger.info("Shutting down ScholarAI API.")


app = FastAPI(
    title="ScholarAI",
    description="RAG-based research assistant API",
    version="0.1.0",
    lifespan=lifespan
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_exception_handler(DocumentNotFoundError, document_not_found_handler)
app.add_exception_handler(RetrievalError, retrieval_error_handler)
app.add_exception_handler(GenerationError, generation_error_handler)
app.add_exception_handler(IndexingError, indexing_error_handler)

app.include_router(index_router.router, prefix="/api/index", tags=["Indexing"])
app.include_router(query_router.router, prefix="/api/query", tags=["Querying"])

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_upload_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}


@app.get("/health", tags=["System"])
def health_check(request: Request):
    pipelines_ready = all([
        getattr(request.app.state, "embedder", None) is not None,
        getattr(request.app.state, "retrieval_pipeline", None) is not None,
        getattr(request.app.state, "generation_pipeline", None) is not None,
        getattr(request.app.state, "indexing_pipeline", None) is not None,
    ])

    status_code = 200 if pipelines_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if pipelines_ready else "degraded",
            "pipelines_loaded": pipelines_ready,
        }
    )