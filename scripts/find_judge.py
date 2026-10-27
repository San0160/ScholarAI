"""
Find which NVIDIA chat models your key can really use as the evaluation judge.

A model can appear in your key's model list and still answer "404 not found",
so this sends each candidate one tiny request and reports what happened.

    python scripts/find_judge.py

Then put the first working name in config/config.yaml under evaluation.judge.model.
Uses NVIDIA_API_KEY and the base URL from evaluation.judge. Costs almost nothing:
each request asks for a few words.
"""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

from ai_research_assistant.config.configuration import ConfigurationManager

# Best judges first: large, instruction-following, not reasoning models, and not
# from the same family as the answer model (OpenAI gpt-oss).
PREFERRED = [
    "mistralai/mistral-large-2-instruct",
    "mistralai/mistral-large",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/llama-3.1-nemotron-51b-instruct",
    "ai21labs/jamba-1.5-large-instruct",
    "databricks/dbrx-instruct",
    "mistralai/mixtral-8x22b-v0.1",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "nvidia/nemotron-3-super-120b-a12b",
    "meta/llama-3.2-90b-vision-instruct",
    "microsoft/phi-3.5-moe-instruct",
    "nv-mistralai/mistral-nemo-12b-instruct",
    "ibm/granite-3.0-8b-instruct",
]
SKIP = ("embed", "guard", "safety", "content", "topic", "retriev", "rerank", "qa", "code", "parse", "clip", "reward")


def main() -> int:
    judge = ConfigurationManager().get_judge_config()
    api_key = os.environ.get(judge.api_key_env_var)
    if not api_key:
        print(f"{judge.api_key_env_var} is not set. Add it to .env.")
        return 1

    client = OpenAI(api_key=api_key, base_url=judge.api_base_url, timeout=30.0, max_retries=0)

    try:
        listed = sorted(model.id for model in client.models.list())
    except Exception as error:
        print(f"Could not list models: {str(error)[:200]}")
        listed = []

    chat_like = [name for name in listed if not any(tag in name.lower() for tag in SKIP)]
    candidates = [name for name in PREFERRED if name in listed or not listed]
    candidates += [name for name in chat_like if name not in candidates]

    print(f"Trying {len(candidates)} model(s) with one tiny request each ...\n")
    working = []
    for name in candidates:
        start = time.perf_counter()
        try:
            reply = client.chat.completions.create(
                model=name,
                messages=[{"role": "user", "content": "Reply with the single word OK."}],
                max_tokens=8,
                temperature=0.0,
            )
            text = (reply.choices[0].message.content or "").strip().replace("\n", " ")
            working.append(name)
            print(f"  WORKS  {name:<52} {time.perf_counter() - start:4.1f}s  reply: {text[:30].encode('ascii', 'replace').decode()}")
        except Exception as error:
            print(f"  fails  {name:<52} {str(error)[:70].encode('ascii', 'replace').decode()}")

    print()
    if working:
        print(f"Use this one: evaluation.judge.model: {working[0]}")
        if len(working) > 1:
            print("Other working options: " + ", ".join(working[1:6]))
        return 0

    print("No model responded. Open https://build.nvidia.com, pick a chat model, click 'Try API' once to enable it for your account, then run this again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
