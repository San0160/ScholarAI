from fastapi import FastAPI

from ai_research_assistant.api.routes import router
from ai_research_assistant.logging import logger


app = FastAPI(
    title="AI Research Assistant",
    version="1.0.0",
    description="AI Research Assistant API"
)

app.include_router(router)

logger.info("Application Started")