import json

from ai_research_assistant.embeddings.embedding_factory import (
    EmbeddingFactory
)
from ai_research_assistant.vector_store.faiss_vector_store import (
    FAISSVectorStore
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

QUESTION_FILE = (
    "src/ai_research_assistant/evaluation/"
    "question_multi_doc.json"
)

STORAGE_PATH = (
    "artifacts/vector_store/multi_document"
)

TOP_K = 5


# --------------------------------------------------
# Load Questions
# --------------------------------------------------

with open(
    QUESTION_FILE,
    "r",
    encoding="utf-8"
) as file:

    test_cases = json.load(file)


# --------------------------------------------------
# Create Embedder
# --------------------------------------------------

embedder = EmbeddingFactory.create_embedding()


# --------------------------------------------------
# Load Vector Store
# --------------------------------------------------

vector_store = FAISSVectorStore(
    dimension=embedder.dimension,
    storage_path=STORAGE_PATH
)

vector_store.load()


# --------------------------------------------------
# Validation Results
# --------------------------------------------------

results = []


# --------------------------------------------------
# Run Validation
# --------------------------------------------------

for index, test_case in enumerate(
    test_cases,
    start=1
):

    question = test_case["question"]

    expected_source = test_case[
        "expected_source"
    ]


    # --------------------------------------------------
    # Embed Query
    # --------------------------------------------------

    query_embedding = embedder.embed_query(
        question
    )


    # --------------------------------------------------
    # Retrieve Results
    # --------------------------------------------------

    retrieved_results = vector_store.similarity_search(
        query_embedding=query_embedding,
        top_k=TOP_K
    )


    # --------------------------------------------------
    # Extract Retrieved Sources
    # --------------------------------------------------

    retrieved_sources = []

    for result in retrieved_results:

        source = result.document.metadata.get(
            "filename"
        )

        if source:

            retrieved_sources.append(source)


    # --------------------------------------------------
    # Check Retrieval Hit
    # --------------------------------------------------

    retrieval_hit = (
        expected_source in retrieved_sources
    )


    # --------------------------------------------------
    # Save Result
    # --------------------------------------------------

    results.append(
        {
            "question": question,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "retrieval_hit": retrieval_hit
        }
    )


    # --------------------------------------------------
    # Display Question Results
    # --------------------------------------------------

    print("\n" + "=" * 80)

    print(
        f"QUESTION {index}"
    )

    print("=" * 80)

    print(
        f"\nQuestion: "
        f"{question}"
    )

    print(
        f"Expected Source: "
        f"{expected_source}"
    )

    print(
        f"Retrieval Hit: "
        f"{retrieval_hit}"
    )

    print(
        "\nRetrieved Results:"
    )


    for rank, result in enumerate(
        retrieved_results,
        start=1
    ):

        document = result.document

        filename = document.metadata.get(
            "filename",
            "unknown"
        )

        print(
            f"\nRank: {rank}"
        )

        print(
            f"Score: "
            f"{result.score:.4f}"
        )

        print(
            f"Source: "
            f"{filename}"
        )

        print(
            f"Chunk: "
            f"{document.metadata.get('chunk_id')}"
        )


# --------------------------------------------------
# Final Evaluation
# --------------------------------------------------

total_questions = len(results)

retrieval_hits = sum(
    result["retrieval_hit"]
    for result in results
)


retrieval_hit_rate = (
    retrieval_hits / total_questions
)


print("\n" + "=" * 80)

print(
    "MULTI-DOCUMENT RETRIEVAL VALIDATION"
)

print("=" * 80)

print(
    f"Questions Tested: "
    f"{total_questions}"
)

print(
    f"Retrieval Hits: "
    f"{retrieval_hits}"
)

print(
    f"Retrieval Hit Rate: "
    f"{retrieval_hit_rate:.4f}"
)