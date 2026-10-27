"""Shared helpers for the evaluation scripts that talk to the judge model."""

import time

from ai_research_assistant.evaluation.llm_judge import LLMJudge
from ai_research_assistant.llm.api_llm import APILLM


def ascii_text(text) -> str:
    """Console-safe text, so a stray symbol cannot crash printing on Windows."""
    return str(text).encode("ascii", "replace").decode("ascii")


def is_rate_limit(error: BaseException) -> bool:
    while error is not None:
        message = str(error).lower()
        if "ratelimit" in type(error).__name__.lower() or "429" in message or "rate limit" in message:
            return True
        error = error.__cause__ or error.__context__
    return False


def with_retries(call, label: str, attempts: int = 4, wait: float = 45.0):
    """Run call(); if the provider says "slow down", wait and try again."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as error:
            if not is_rate_limit(error) or attempt == attempts:
                raise
            print(f"    {label}: rate limited, waiting {wait:.0f}s (retry {attempt} of {attempts - 1})")
            time.sleep(wait)


def make_judge(config_manager) -> LLMJudge:
    """Build the judge and test it with one tiny call, so a wrong model name or
    key fails now instead of after twenty minutes."""
    judge_config = config_manager.get_judge_config()
    llm = APILLM(
        model=judge_config.model,
        base_url=judge_config.api_base_url,
        api_key_env_var=judge_config.api_key_env_var,
        max_new_tokens=judge_config.max_new_tokens,
    )
    try:
        llm.generate([{"role": "user", "content": "Reply with the single word OK."}])
    except Exception as error:
        print(f"\nThe judge model '{judge_config.model}' did not respond: {ascii_text(error.__cause__ or error)}")
        print("Run 'python scripts/find_judge.py' to see which models your key can really use.")
        raise SystemExit(2)
    return LLMJudge(llm)
