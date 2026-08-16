from ai_research_assistant.generation.prompt_builder import PromptBuilder


builder = PromptBuilder()

query = (
    "What optimization algorithm and learning-rate schedule "
    "were used to train the Transformer?"
)

context = """
The big models were trained for 300,000 steps.
We used the Adam optimizer with β1 = 0.9,
β2 = 0.98 and ε = 10−9.
We varied the learning rate over the course of training.
This corresponds to increasing the learning rate linearly
for the first warmup steps and decreasing it thereafter
proportionally to the inverse square root of the step number.
"""

prompt = builder.build(
    query=query,
    context=context
)

print("=" * 80)
print("GENERATED PROMPT")
print("=" * 80)
print(prompt)