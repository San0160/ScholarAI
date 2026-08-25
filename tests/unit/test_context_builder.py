from ai_research_assistant.generation.context_builder import ContextBuilder
from ai_research_assistant.entity.document import Document


documents = [

    Document(
        page_content=(
            "We used the Adam optimizer with "
            "β1 = 0.9, β2 = 0.98 and ε = 10−9."
        ),
        metadata={
            "filename": "attention.pdf",
            "page": 7
        }
    ),

    Document(
        page_content=(
            "The learning rate was increased linearly "
            "during the warmup period."
        ),
        metadata={
            "filename": "attention.pdf",
            "page": 7
        }
    )
]


builder = ContextBuilder()

context = builder.build(
    documents
)


print("=" * 80)
print("BUILT CONTEXT")
print("=" * 80)
print(context)