import logging

import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ai_research_assistant.llm.base_llm import BaseLLM
from ai_research_assistant.utils.device import resolve_device

logger = logging.getLogger(__name__)

class HuggingFaceLLM(BaseLLM):

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        max_context_tokens: int = 1024,
        max_new_tokens: int = 256
    ):

        self.device = torch.device(resolve_device(device))

        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens

        logger.info("Loading LLM '%s' on device '%s'", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            low_cpu_mem_usage=True
        )

        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        messages: list[dict]
    ) -> str:

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        prompt_tokens = inputs["input_ids"].shape[1]

        if prompt_tokens > self.max_context_tokens:
            logger.warning(
                "Prompt is %d tokens, exceeding max_context_tokens=%d -- "
                "generation may be slower than expected",
                prompt_tokens, self.max_context_tokens,
            )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        start = time.perf_counter()

        try:
            with torch.no_grad():

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id
                )

        except Exception as error:
            raise RuntimeError(
                f"LLM generation failed after {prompt_tokens} prompt token(s) "
                f"on device '{self.device}'"
            ) from error

        elapsed = time.perf_counter() - start

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        completion_tokens = generated_tokens.shape[0]
        rate = completion_tokens / elapsed if elapsed > 0 else float("inf")

        logger.info(
            "Generated %d token(s) from %d prompt token(s) in %.2fs (%.1f tok/s)",
            completion_tokens, prompt_tokens, elapsed, rate,
        )

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return answer.strip()