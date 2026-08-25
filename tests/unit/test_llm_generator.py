from ai_research_assistant.llm.llm_factory import LLMFactory
from ai_research_assistant.generation.llm_generator import LLMGenerator


llm = LLMFactory.create_llm()
generator = LLMGenerator(llm=llm)


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

messages = [
    {
        "role": "system",
        "content": (
            "You are ScholarAI, a research assistant. "
            "Answer only using the provided context."
        )
    },
    {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{query}"
    }
]

answer = generator.generate(messages)

print("=" * 80)
print("GENERATED ANSWER")
print("=" * 80)
print(answer)