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

        index_tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
        documents_tmp_path = documents_path.with_suffix(documents_path.suffix + ".tmp")

        faiss.write_index(self.index, str(index_tmp_path))

        with open(documents_tmp_path, "wb") as file:
            pickle.dump(self.documents, file)

        index_tmp_path.replace(index_path)
        documents_tmp_path.replace(documents_path)

        logger.info(
            "Saved vector store: %d vectors, %d documents -> %s",
            self.index.ntotal,
            len(self.documents),
            self.storage_path,
        )

    def load(self):
        index_path = self.storage_path / self.index_name
        documents_path = self.storage_path / self.documents_name

        if not index_path.exists() or not documents_path.exists():
            raise FileNotFoundError(
                f"No existing vector store found at '{self.storage_path}' "
                f"(expected '{self.index_name}' and '{self.documents_name}')"
            )

        index = faiss.read_index(str(index_path))

        if index.d != self.dimension:
            raise ValueError(
                f"Loaded index dimension ({index.d}) does not match the configured "
                f"embedding dimension ({self.dimension}) -- this usually means the embedding "
                f"model/provider changed since this index was built. Delete the existing "
                f"index/documents files (or run indexing with rebuild=True) to re-embed "
                f"everything with the current model."
            )

        with open(documents_path, "rb") as f:
            documents = pickle.load(f)

        if index.ntotal != len(documents):
            raise ValueError(
                f"Index/documents count mismatch after loading: index has {index.ntotal} "
                f"vector(s) but {len(documents)} document(s) were loaded -- the vector store "
                f"files may be corrupted or out of sync."
            )

        self.index = index
        self.documents = documents
        logger.info(
            "Loaded vector store from '%s': %d document(s), dimension=%d",
            self.storage_path, len(self.documents), self.dimension,
        )

    def clear(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []

        logger.info("Cleared vector store (dimension=%d)", self.dimension)