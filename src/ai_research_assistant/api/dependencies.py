from fastapi import Request

from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.pipeline.generation_pipeline import GenerationPipeline
from ai_research_assistant.pipeline.indexing_pipeline import IndexingPipeline


def get_retrieval_pipeline(request: Request) -> RetrievalPipeline:
    return request.app.state.retrieval_pipeline


def get_generation_pipeline(request: Request) -> GenerationPipeline:
    return request.app.state.generation_pipeline


def get_indexing_pipeline(request: Request) -> IndexingPipeline:
    return request.app.state.indexing_pipeline