import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from ai_research_assistant.entity.document import Document
from ai_research_assistant.entity.retrieval_result import RetrievalResult
from ai_research_assistant.vector_store.base_vector import BaseVectorStore

logger = logging.getLogger(__name__)

_NORM_TOLERANCE = 1e-3


class FAISSVectorStore(BaseVectorStore):

    def __init__(
        self,
        dimension: int,
        storage_path: str,
        index_name: str = "index.faiss",
        documents_name: str = "documents.pkl",
    ):

        self.dimension = dimension
        self.storage_path = Path(storage_path)
        self.index_name = index_name
        self.documents_name = documents_name

        self.storage_path.mkdir(
            parents=True,
            exist_ok=True
        )

        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]]
    ):
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents and embeddings must be the same length, got "
                f"{len(documents)} documents and {len(embeddings)} embeddings"
            )

        if not documents:
            return

        vectors = np.array(
            embeddings,
            dtype="float32"
        )

        self._validate_vectors(vectors, context="add_documents")

        self.index.add(vectors)

        self.documents.extend(documents)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[RetrievalResult]:

        query_vector = np.array(
            [query_embedding],
            dtype="float32"
        )

        self._validate_vectors(query_vector, context="similarity_search query")

        scores, indices = self.index.search(
            query_vector,
            top_k
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index == -1:
                continue

            results.append(
                RetrievalResult(
                    document=self.documents[index],
                    score=float(score)
                )
            )

        return results

    def _validate_vectors(self, vectors: np.ndarray, context: str) -> None:

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"{context}: expected vectors of dimension {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        norms = np.linalg.norm(vectors, axis=1)
        not_normalized = np.abs(norms - 1.0) > _NORM_TOLERANCE

        if np.any(not_normalized):
            logger.warning(
                "%s: %d/%d vectors are not unit-normalized -- IndexFlatIP's "
                "inner product will not equal cosine similarity, so "
                "retrieval ranking will not be meaningful",
                context,
                int(np.sum(not_normalized)),
                len(vectors),
            )

    def save(self):

        index_path = self.storage_path / self.index_name
        documents_path = self.storage_path / self.documents_name

        faiss.write_index(
            self.index,
            str(index_path)
        )

        with open(documents_path, "wb") as file:
            pickle.dump(self.documents, file)

    def load(self):

        index_path = self.storage_path / self.index_name
        documents_path = self.storage_path / self.documents_name

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")

        if not documents_path.exists():
            raise FileNotFoundError(f"Documents file not found: {documents_path}")

        self.index = faiss.read_index(str(index_path))

        with open(documents_path, "rb") as file:
            self.documents = pickle.load(file)