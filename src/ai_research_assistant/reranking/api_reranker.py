# reranking/api_reranker.py
import logging
import os

import requests

from ai_research_assistant.entity.retrieval_result import RetrievalResult
from ai_research_assistant.reranking.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class APIReranker(BaseReranker):
    """Hosted reranking backend for NVIDIA NIM's /reranking endpoint. Unlike
    APILLM/APIEmbedding, this does NOT reuse the openai client -- NVIDIA's
    reranking API has its own request/response shape (query + passages in,
    rankings with index/logit out), not an OpenAI-compatible one.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env_var: str,
        timeout: float = 30.0,
    ):
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{api_key_env_var}' is not set -- add it to your .env file"
            )

        self.model = model
        self.base_url = base_url  # full endpoint URL, not just a host prefix
        self.timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        logger.info("Configured API reranker backend: model='%s', base_url='%s'", model, base_url)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int
    ) -> list[RetrievalResult]:

        if not results:
            return []

        payload = {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": result.document.page_content} for result in results],
            "truncate": "END",
        }

        try:
            response = requests.post(
                self.base_url,
                headers=self._headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as error:
            raise RuntimeError(f"API reranker request failed for model '{self.model}'") from error

        rankings = response.json().get("rankings", [])

        reranked_results = [
            RetrievalResult(
                document=results[entry["index"]].document,
                score=float(entry["logit"]),
            )
            for entry in rankings
            if entry.get("index") is not None and entry["index"] < len(results)
        ]

        return reranked_results[:top_k]