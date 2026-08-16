import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ai_research_assistant.generation.base_generator import BaseGenerator


class LLMGenerator(BaseGenerator):

    def __init__(
        self,
        model_name: str,
        max_context_tokens: int = 1024
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.max_context_tokens = max_context_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype
        )

        self.model.to(
            self.device
        )

        self.model.eval()

    def _truncate_context(
        self,
        context: str
    ) -> str:

        tokens = self.tokenizer.encode(
            context,
            add_special_tokens=False
        )

        if len(tokens) > self.max_context_tokens:

            tokens = tokens[
                :self.max_context_tokens
            ]

            context = self.tokenizer.decode(
                tokens
            )

        return context

    def generate(
        self,
        messages: list[dict]
    ) -> str:

        if not messages:

            return (
                "The information is not available "
                "in the provided document."
            )

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated_tokens = outputs[
            0
        ][
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return answer.strip()