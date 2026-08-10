from transformers import AutoTokenizer, AutoModelForCausalLM

from ai_research_assistant.llm.base_llm import BaseLLM


class HuggingFaceLLM(BaseLLM):

    def __init__(self, model_name: str):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

    def generate(
        self,
        prompt: str
    ) -> str:

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2
        )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return response