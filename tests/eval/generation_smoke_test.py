from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.pipeline.generation_pipeline import GenerationPipeline
from ai_research_assistant.entity.document import Document


embedder = EmbeddingFactory.create_embedding()

pipeline = GenerationPipeline(embedder=embedder)


query = (
    "What optimization algorithm and learning-rate schedule "
    "were used to train the Transformer?"
)

documents = [
    Document(
        page_content=(
            "We used the Adam optimizer with "
            "β1 = 0.9, β2 = 0.98 and ε = 10−9. "
            "We varied the learning rate over the course "
            "of training."
        ),
        metadata={"filename": "attention.pdf", "page": 7, "chunk_id": "manual_1"}
    ),
    Document(
        page_content=(
            "This corresponds to increasing the learning rate "
            "linearly for the first warmup_steps training steps, "
            "and decreasing it thereafter proportionally to the "
            "inverse square root of the step number."
        ),
        metadata={"filename": "attention.pdf", "page": 7, "chunk_id": "manual_2"}
    )
]

result = pipeline.run(query=query, documents=documents)

print("=" * 80)
print("FORMATTED ANSWER")
print("=" * 80)
print(result["answer"])

print("\n" + "=" * 80)
print("RAW ANSWER (pre-formatting)")
print("=" * 80)
print(result["raw_answer"])