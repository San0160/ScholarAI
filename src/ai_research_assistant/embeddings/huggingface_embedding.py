from sentence_transformers import SentenceTransformer
from ai_research_assistant.embeddings.base_embedding import BaseEmbedding


class HuggingFaceEmbedding(BaseEmbedding):

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist() 