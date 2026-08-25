import json

from ai_research_assistant.embeddings.embedding_factory import (
    EmbeddingFactory
)
from ai_research_assistant.vector_store.faiss_vector_store import (
    FAISSVectorStore
)
from ai_research_assistant.pipeline.generation_pipeline import (
    GenerationPipeline
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
# Create Generation Pipeline
# --------------------------------------------------

generation_pipeline = GenerationPipeline(
    embedder=embedder
)


# --------------------------------------------------
# Run Multi-Document Generation
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
    # Retrieve Documents
    # --------------------------------------------------

    retrieved_results = vector_store.similarity_search(
        query_embedding=query_embedding,
        top_k=TOP_K
    )


    documents = [
        result.document
        for result in retrieved_results
    ]


    # --------------------------------------------------
    # Debug Retrieved Context
    # --------------------------------------------------

    if index == 3:

        print("\n" + "=" * 80)

        print(
            "Q3 RETRIEVED CONTEXT DEBUG"
        )

        print("=" * 80)

        for rank, result in enumerate(
            retrieved_results,
            start=1
        ):

            document = result.document

            print(
                f"\nRANK {rank}"
            )

            print("-" * 80)

            print(
                f"Source: "
                f"{document.metadata.get('filename', 'unknown')}"
            )

            print(
                f"Chunk: "
                f"{document.metadata.get('chunk_id', 'unknown')}"
            )

            print(
                f"Score: "
                f"{result.score:.4f}"
            )

            print(
                "\nFULL CHUNK CONTENT:\n"
            )

            print(
                document.page_content
            )


    # --------------------------------------------------
    # Generate Answer
    # --------------------------------------------------

    generation_result = generation_pipeline.run(
        query=question,
        documents=documents
    )


    generated_answer = generation_result[
        "answer"
    ]


    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------

    print("\n" + "=" * 80)

    print(
        f"QUESTION {index}"
    )

    print("=" * 80)

    print(
        f"\nQuestion: {question}"
    )

    print(
        f"\nExpected Source: {expected_source}"
    )

    print(
        "\nGENERATED ANSWER"
    )

    print("-" * 80)

    print(
        generated_answer
    )


    # --------------------------------------------------
    # Retrieved Sources
    # --------------------------------------------------

    print(
        "\nRETRIEVED SOURCES"
    )

    print("-" * 80)


    for rank, result in enumerate(
        retrieved_results,
        start=1
    ):

        document = result.document

        print(
            f"Rank {rank} | "
            f"Score: {result.score:.4f} | "
            f"Source: "
            f"{document.metadata.get('filename', 'unknown')} | "
            f"Chunk: "
            f"{document.metadata.get('chunk_id', 'unknown')}"
        )