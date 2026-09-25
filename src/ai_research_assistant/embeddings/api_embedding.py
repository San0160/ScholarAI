import logging
import os
import time
import numpy as np

from openai import OpenAI

from ai_research_assistant.embeddings.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)


class APIEmbedding(BaseEmbedding):
    """Hosted embedding backend for an OpenAI-compatible /v1/embeddings API
    that distinguishes query vs. passage encoding via `input_type` (NVIDIA
    NIM's NeMo Retriever embedding models, among others). Passing the wrong
    input_type is the same class of silent-mismatch bug already fixed in
    CitationMatcher's embed_query/embed_documents asymmetry -- just at the
    API layer instead of inside a local model.
    """

    def __init__(self, model: str, base_url: str, api_key_env_var: str, dimension: int, timeout: float = 60.0):
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{api_key_env_var}' is not set -- add it to your .env file"
            )
        self._dimension = dimension
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        logger.info("Configured API embedding backend: model='%s', base_url='%s', dimension=%d", model, base_url, dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="query")[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        start = time.perf_counter()
        embeddings = self._embed(texts, input_type="passage")
        elapsed = time.perf_counter() - start
        rate = len(texts) / elapsed if elapsed > 0 else float("inf")
        logger.info("Embedded %d document(s) in %.2fs (%.1f docs/sec)", len(texts), elapsed, rate)
        return embeddings

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self._dimension,
                extra_body={"input_type": input_type},
            )
        except Exception as error:
            raise RuntimeError(
                f"API embedding request failed for model '{self.model}' (input_type='{input_type}')"
            ) from error

        vectors = np.array([item.embedding for item in response.data], dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid divide-by-zero on a degenerate all-zero embedding
        return (vectors / norms).tolist()