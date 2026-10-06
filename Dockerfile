FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only PyTorch first. A plain "pip install torch" on Linux pulls the GPU
# build, which adds several GB to the image for hardware the server doesn't have.
# requirements.txt leaves torch unpinned, so pip keeps this build afterwards.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY . .

# Editable install keeps the package under /app/src, which constants.py relies on
# to locate config/, data/ and logs/ relative to the project root.
RUN pip install -r requirements.txt && pip install -e .

# OPTIONAL: download the embedding model and the tokenizer into the image, so the
# app doesn't fetch them from Hugging Face on every start. Skipped automatically
# when embeddings use an API provider. If this step fails the build, delete it:
# the app still works, it just starts slower.
RUN python -c "from ai_research_assistant.config.configuration import ConfigurationManager as C; from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory as E; from ai_research_assistant.utils.tokenizer_utils import get_token_counter as t; m = C(); e = m.get_embedding_config(); e.provider == 'huggingface' and E.create_embedding(e); t(m.get_llm_config().model)('warm up')"

EXPOSE 8000

# One worker only: the vector index lives in this process's memory.
# Railway supplies PORT; 8000 is the fallback for running the image locally.
CMD ["sh", "-c", "exec uvicorn ai_research_assistant.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]