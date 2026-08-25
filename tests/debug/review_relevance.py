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
# Vector Store
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
    top_k=10
)


# --------------------------------------------------
# Review chunks
# --------------------------------------------------

for i, item in enumerate(
    questions,
    start=1
):

    if i not in [6, 7, 8, 9, 13, 16]:
        continue
    
    query = item["question"]

    print("\n" + "=" * 100)
    print(f"QUESTION {i}: {query}")
    print("=" * 100)

    print(
        f"Expected chunks: "
        f"{item['expected_chunks']}"
    )

    results = retriever.retrieve(
        query
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        chunk_id = result.document.metadata.get(
            "chunk_id"
        )

        page = result.document.metadata.get(
            "page"
        )

        print("\n" + "-" * 100)

        print(
            f"Rank: {rank}"
        )

        print(
            f"Chunk: {chunk_id}"
        )

        print(
            f"Page: {page}"
        )

        print(
            f"Score: {result.score}"
        )

        print(
            "Content:"
        )

        print(
            result.document.page_content
        )