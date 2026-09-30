import logging
import os

from openai import OpenAI

from ai_research_assistant.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)


class APILLM(BaseLLM):
    """Hosted chat-completions backend for any OpenAI-compatible
    /v1/chat/completions API (Groq, OpenRouter, NVIDIA NIM, etc.) --
    swapping providers is a base_url/api_key_env_var/model change in
    config.yaml.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env_var: str,
        max_new_tokens: int = 256,
        timeout: float = 60.0,
    ):
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{api_key_env_var}' is not set -- add it to your .env file"
            )

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        logger.info("Configured API LLM backend: model='%s', base_url='%s'", model, base_url)

    def generate(self, messages: list[dict]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=0.0,
            )
        except Exception as error:
            raise RuntimeError(f"API LLM request failed for model '{self.model}'") from error

        answer = response.choices[0].message.content
        return (answer or "").strip()