import faiss
import numpy as np
import pickle
from pathlib import Path

from ai_research_assistant.entity.document import Document
from ai_research_assistant.vector_store.base_vector import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):

    def __init__(
        self,
        dimension: int,
        storage_path: str
    ):

        self.dimension = dimension
        self.storage_path = Path(storage_path)

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

        vectors = np.array(
            embeddings,
            dtype="float32"
        )

        self.index.add(vectors)

        self.documents.extend(documents)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[Document]:

        query_vector = np.array(
            [query_embedding],
            dtype="float32"
        )

        scores, indices = self.index.search(
            query_vector,
            top_k
        )

        results = []

        for index in indices[0]:

            if index == -1:
                continue

            results.append(
                self.documents[index]
            )

        return results

    def save(self):

        index_path = self.storage_path / "scholarai.index"
        documents_path = self.storage_path / "documents.pkl"

        faiss.write_index(
            self.index,
            str(index_path)
        )

        with open(
            documents_path,
            "wb"
        ) as file:

            pickle.dump(
                self.documents,
                file
            )

    def load(self):

        index_path = self.storage_path / "scholarai.index"
        documents_path = self.storage_path / "documents.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not documents_path.exists():
            raise FileNotFoundError(
                f"Documents file not found: {documents_path}"
            )

        self.index = faiss.read_index(
            str(index_path)
        )

        with open(
            documents_path,
            "rb"
        ) as file:

            self.documents = pickle.load(file)