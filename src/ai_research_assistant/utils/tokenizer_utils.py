# ai_research_assistant/chunking/tokenizer_utils.py
from functools import lru_cache
from typing import Callable

from transformers import AutoTokenizer


@lru_cache(maxsize=4)
def _load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def get_token_counter(model_name: str) -> Callable[[str], int]:
    """Token-counting function bound to `model_name`'s tokenizer."""
    tokenizer = _load_tokenizer(model_name)

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count_tokens