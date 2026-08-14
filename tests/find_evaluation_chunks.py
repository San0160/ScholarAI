import json

from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.vector_store.vector_store_factory import VectorStoreFactory
from ai_research_assistant.retrieval.vector_retriver import VectorRetriever


# --------------------------------------------------
# Load evaluation questions
# --------------------------------------------------

with open(
    "src/ai_research_assistant/evaluation/question_attention.json",
    "r",
    encoding="utf-8"
) as file:

    questions = json.load(file)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

config = ConfigurationManager().config


# --------------------------------------------------
# Embedding
# --------------------------------------------------

embedder = EmbeddingFactory.create_embedding()


# --------------------------------------------------
# Vector store
# --------------------------------------------------

vector_store = VectorStoreFactory.create_vector_store(
    dimension=embedder.dimension
)

vector_store.load()


# --------------------------------------------------
# Retriever
# --------------------------------------------------

retriever = VectorRetriever(
    embedder=embedder,
    vector_store=vector_store,
    top_k=5
)


# --------------------------------------------------
# Search each question
# --------------------------------------------------

for number, item in enumerate(
    questions,
    start=1
):

    question = item["question"]

    results = retriever.retrieve(
        question
    )

    print("\n" + "=" * 100)

    print(
        f"QUESTION {number}: {question}"
    )

    print("=" * 100)

    for rank, result in enumerate(
        results,
        start=1
    ):

        metadata = result.document.metadata

        content = (
            result.document.page_content
            .replace("\n", " ")
        )

        print(
            f"\nRank: {rank}"
        )

        print(
            "Chunk:",
            metadata.get("chunk_id")
        )

        print(
            "Page:",
            metadata.get("page")
        )

        print(
            "Score:",
            result.score
        )

        print(
            "Content:",
            content[:500]
        )