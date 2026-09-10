import logging
import time

from sentence_transformers import SentenceTransformer

from ai_research_assistant.embeddings.base_embedding import BaseEmbedding
from ai_research_assistant.utils.device import resolve_device

logger = logging.getLogger(__name__)

class HuggingFaceEmbedding(BaseEmbedding):

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ):
        resolved_device = resolve_device(device)
        logger.info(
            "Loading embedding model '%s' on device '%s'", model_name, resolved_device
        )

        self.model = SentenceTransformer(model_name, device=resolved_device)
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

    @property
    def dimension(self) -> int:

        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:

        self._warn_if_truncated(texts)

        logger.info("Embedding %d documents (batch_size=%d)", len(texts), self.batch_size)
        start = time.perf_counter()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=True
        )

        elapsed = time.perf_counter() - start
        rate = len(texts) / elapsed if elapsed > 0 else float("inf")
        logger.info(
            "Embedded %d documents in %.2fs (%.1f docs/sec)", len(texts), elapsed, rate
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:

        self._warn_if_truncated([text])

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()


    def _warn_if_truncated(self, texts: list[str]) -> None:

        """sentence-transformers silently truncates any text longer than
        max_seq_length ***** this surfaces that instead of losing content
        without a trace.
        """

        max_len = self.model.max_seq_length

        over_limit = [
            (index, len(self.model.tokenizer.encode(text)))
            for index, text in enumerate(texts)
        ]
        over_limit = [
            (index, length) for index, length in over_limit if length > max_len
        ]

        if not over_limit:
            return

        worst_index, worst_length = max(over_limit, key=lambda pair: pair[1])

        logger.warning(
            "%d/%d texts exceed max_seq_length=%d and will be silently "
            "truncated by sentence-transformers (worst case: text #%d at "
            "%d tokens)",
            len(over_limit),
            len(texts),
            max_len,
            worst_index,
            worst_length,
        )