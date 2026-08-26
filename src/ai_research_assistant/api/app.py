from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.pipeline.generation_pipeline import GenerationPipeline
from ai_research_assistant.pipeline.indexing_pipeline import IndexingPipeline
from ai_research_assistant.api.routers import index_router, query_router
from ai_research_assistant.logging.logger import logger

from ai_research_assistant.api.exceptions import (
    DocumentNotFoundError, RetrievalError,
    document_not_found_handler, retrieval_error_handler
)



@asynccontextmanager
async def lifespan(app: FastAPI):

    # --- Startup: load everything once ---
    logger.info("Loading models and pipelines...")

    embedder = EmbeddingFactory.create_embedding()

    app.state.embedder = embedder
    app.state.retrieval_pipeline = RetrievalPipeline()
    app.state.generation_pipeline = GenerationPipeline(embedder=embedder)
    app.state.indexing_pipeline = IndexingPipeline()

    logger.info("Startup complete.")

    yield  # app runs here

    # --- Shutdown: cleanup if needed ---
    logger.info("Shutting down ScholarAI API.")


app = FastAPI(
    title="ScholarAI",
    description="RAG-based research assistant API",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(index_router.router, prefix="/index", tags=["Indexing"])
app.include_router(query_router.router, prefix="/query", tags=["Querying"])
app.add_exception_handler(DocumentNotFoundError, document_not_found_handler)
app.add_exception_handler(RetrievalError, retrieval_error_handler)


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}