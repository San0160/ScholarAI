from ai_research_assistant.config.configuration import ConfigurationManager

from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.pipeline.generation_pipeline import GenerationPipeline


# --------------------------------------------------
# Configuration
# --------------------------------------------------

config = ConfigurationManager().config


# --------------------------------------------------
# Question
# --------------------------------------------------

query = (
    "What optimization algorithm and learning-rate schedule "
    "were used to train the Transformer?"
)


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

retrieval_pipeline = RetrievalPipeline()

retrieved_results = retrieval_pipeline.run(
    query
)


# --------------------------------------------------
# Display Retrieved Chunks
# --------------------------------------------------

print("\n" + "=" * 80)
print("RETRIEVED CONTEXT")
print("=" * 80)

for rank, result in enumerate(
    retrieved_results,
    start=1
):

    chunk_id = result.document.metadata.get(
        "chunk_id",
        "unknown"
    )

    page = result.document.metadata.get(
        "page",
        "unknown"
    )

    print(
        f"\nRank: {rank}"
        f"\nChunk: {chunk_id}"
        f"\nPage: {page}"
        f"\nScore: {result.score}"
    )

    print(
        f"\nContent:\n"
        f"{result.document.page_content}"
    )


# --------------------------------------------------
# Extract Documents
# --------------------------------------------------

documents = [
    result.document
    for result in retrieved_results
]


# --------------------------------------------------
# Generation
# --------------------------------------------------

generation_pipeline = GenerationPipeline(
    model_name="Qwen/Qwen2.5-1.5B-Instruct"
)


answer = generation_pipeline.run(
    query=query,
    documents=documents
)


# --------------------------------------------------
# Final Answer
# --------------------------------------------------

print("\n" + "=" * 80)
print("SCHOLARAI ANSWER")
print("=" * 80)

print(answer)