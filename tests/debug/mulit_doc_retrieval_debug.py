from ai_research_assistant.embeddings.embedding_factory import (
    EmbeddingFactory
)
from ai_research_assistant.vector_store.faiss_vector_store import (
    FAISSVectorStore
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

STORAGE_PATH = (
    "artifacts/vector_store/multi_document"
)

QUERY = (
    "What optimization algorithm was used "
    "to train the Transformer?"
)

TOP_K = 5


# --------------------------------------------------
# Create Embedder
# --------------------------------------------------

embedder = EmbeddingFactory.create_embedding()


# --------------------------------------------------
# Load Multi-Document Vector Store
# --------------------------------------------------

vector_store = FAISSVectorStore(
    dimension=embedder.dimension,
    storage_path=STORAGE_PATH
)

vector_store.load()


# --------------------------------------------------
# Display Loaded Index
# --------------------------------------------------

print("\n" + "=" * 80)
print("MULTI-DOCUMENT RETRIEVAL")
print("=" * 80)

print(
    f"Vectors Loaded: "
    f"{vector_store.index.ntotal}"
)

print(
    f"Documents Loaded: "
    f"{len(vector_store.documents)}"
)


# --------------------------------------------------
# Embed Query
# --------------------------------------------------

query_embedding = embedder.embed_query(
    QUERY
)


# --------------------------------------------------
# Retrieve Results
# --------------------------------------------------

results = vector_store.similarity_search(
    query_embedding=query_embedding,
    top_k=TOP_K
)


# --------------------------------------------------
# Display Results
# --------------------------------------------------

print(f"\nQuery: {QUERY}")

print(
    f"\nRetrieved Results: "
    f"{len(results)}"
)


for rank, result in enumerate(
    results,
    start=1
):

    document = result.document

    print("\n" + "-" * 80)

    print(
        f"Rank: {rank}"
    )

    print(
        f"Score: "
        f"{result.score:.4f}"
    )

    print(
        f"Metadata: "
        f"{document.metadata}"
    )

    print(
        "\nContent Preview:"
    )

    print(
        document.page_content[:500]
    )