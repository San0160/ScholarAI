from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.pipeline.generation_pipeline import GenerationPipeline
from ai_research_assistant.pipeline.indexing_pipeline import IndexingPipeline
from ai_research_assistant.api.routers import index_router, query_router
from ai_research_assistant.logging.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Loading models and pipelines...")

    embedder = EmbeddingFactory.create_embedding()

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

app.include_router(index_router.router, prefix="/api/index", tags=["Indexing"])
app.include_router(query_router.router, prefix="/api/query", tags=["Querying"])

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def serve_upload_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}