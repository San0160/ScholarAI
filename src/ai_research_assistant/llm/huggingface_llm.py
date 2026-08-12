from transformers import AutoTokenizer, AutoModelForCausalLM
from ai_research_assistant.llm.base_llm import BaseLLM
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

class HuggingFaceLLM(BaseLLM):

    def __init__(self, model_name: str):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )

        self.model.eval()

    def generate(self, prompt: str) -> str:

        messages = [
            {
                "role": "system",
                "content": (
                    "You are ScholarAI, an AI research assistant. "
                    "Answer only using the provided context. "
                    "Do not invent information."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

        input_length = inputs["input_ids"].shape[1]

        generated_tokens = outputs[0][input_length:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return answer.strip()