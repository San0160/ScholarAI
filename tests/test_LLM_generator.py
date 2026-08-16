from ai_research_assistant.generation.llm_generator import LLMGenerator


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


generator = LLMGenerator(
    model_name=MODEL_NAME
)


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
for the first warmup_steps training steps, and decreasing it
thereafter proportionally to the inverse square root of
the step number.
"""


answer = generator.generate(
    query=query,
    context=context
)


print("=" * 80)
print("GENERATED ANSWER")
print("=" * 80)
print(answer)