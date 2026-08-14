from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.vector_store.vector_store_factory import VectorStoreFactory
from ai_research_assistant.retrieval.vector_retriver import VectorRetriever
from ai_research_assistant.retrieval.metadata_filter import MetadataFilter
from ai_research_assistant.reranking.cross_encoder_reranker import CrossEncoderReranker


query = (
    "What optimization algorithm and learning-rate schedule "
    "were used to train the Transformer?"
)


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
    top_k=config.retrieval.candidate_k
)


# --------------------------------------------------
# Metadata Filter
# --------------------------------------------------

metadata_filter = MetadataFilter()


# --------------------------------------------------
# Stage 1: FAISS Retrieval
# --------------------------------------------------

candidates = retriever.retrieve(
    query
)


print("\n" + "=" * 100)
print("FAISS RETRIEVAL")
print("=" * 100)


for rank, result in enumerate(
    candidates,
    start=1
):

    chunk_id = result.document.metadata.get(
        "chunk_id"
    )

    page = result.document.metadata.get(
        "page"
    )

    print(
        f"FAISS Rank: {rank} | "
        f"Chunk: {chunk_id} | "
        f"Page: {page} | "
        f"FAISS Score: {result.score}"
    )


# --------------------------------------------------
# Stage 2: Metadata Filter
# --------------------------------------------------

candidates = metadata_filter.filter(
    candidates
)


# --------------------------------------------------
# Save FAISS ranking
# --------------------------------------------------

faiss_ranking = {}

for rank, result in enumerate(
    candidates,
    start=1
):

    chunk_id = result.document.metadata.get(
        "chunk_id"
    )

    faiss_ranking[chunk_id] = {
        "rank": rank,
        "score": result.score
    }


# --------------------------------------------------
# Stage 3: Cross-Encoder
# --------------------------------------------------

reranker = CrossEncoderReranker(
    model_name=config.reranking.model
)


# --------------------------------------------------
# Calculate reranker scores manually
# --------------------------------------------------

pairs = [
    (
        query,
        result.document.page_content
    )
    for result in candidates
]


scores = reranker.model.predict(
    pairs
)


reranker_results = []

for result, score in zip(
    candidates,
    scores
):

    chunk_id = result.document.metadata.get(
        "chunk_id"
    )

    reranker_results.append(
        {
            "chunk_id": chunk_id,
            "page": result.document.metadata.get(
                "page"
            ),
            "faiss_rank": faiss_ranking[
                chunk_id
            ]["rank"],
            "faiss_score": faiss_ranking[
                chunk_id
            ]["score"],
            "reranker_score": float(score),
            "document": result.document
        }
    )


# --------------------------------------------------
# Sort by reranker score
# --------------------------------------------------

reranker_results.sort(
    key=lambda item: item["reranker_score"],
    reverse=True
)


# --------------------------------------------------
# Final comparison
# --------------------------------------------------

print("\n" + "=" * 100)
print("FAISS vs RERANKER")
print("=" * 100)


for reranker_rank, item in enumerate(
    reranker_results,
    start=1
):

    print(
        f"Reranker Rank: {reranker_rank} | "
        f"FAISS Rank: {item['faiss_rank']} | "
        f"Chunk: {item['chunk_id']} | "
        f"Page: {item['page']} | "
        f"FAISS Score: {item['faiss_score']:.6f} | "
        f"Reranker Score: {item['reranker_score']:.6f}"
    )


# --------------------------------------------------
# Top 3
# --------------------------------------------------

print("\n" + "=" * 100)
print("FINAL RERANKER TOP 3")
print("=" * 100)


for rank, item in enumerate(
    reranker_results[:config.reranking.top_k],
    start=1
):

    print(
        f"Rank: {rank} | "
        f"Chunk: {item['chunk_id']} | "
        f"Page: {item['page']} | "
        f"FAISS Rank: {item['faiss_rank']} | "
        f"FAISS Score: {item['faiss_score']:.6f} | "
        f"Reranker Score: {item['reranker_score']:.6f}"
    )