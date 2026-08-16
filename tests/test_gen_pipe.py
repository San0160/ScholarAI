from ai_research_assistant.pipeline.generation_pipeline import (
    GenerationPipeline
)

from ai_research_assistant.entity.document import Document


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


pipeline = GenerationPipeline(
    model_name=MODEL_NAME
)


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
        metadata={
            "filename": "attention.pdf",
            "page": 7
        }
    ),

    Document(
        page_content=(
            "This corresponds to increasing the learning rate "
            "linearly for the first warmup_steps training steps, "
            "and decreasing it thereafter proportionally to the "
            "inverse square root of the step number."
        ),
        metadata={
            "filename": "attention.pdf",
            "page": 7
        }
    )
]


answer = pipeline.run(
    query=query,
    documents=documents
)


print("=" * 80)
print("GENERATED ANSWER")
print("=" * 80)
print(answer)