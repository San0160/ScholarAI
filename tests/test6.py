"""
Standalone sanity check for the NVIDIA NIM reranking API -- run this BEFORE
trusting APIReranker inside the full RetrievalPipeline.

It sends one query + a few passages directly to the hosted endpoint and
prints the raw JSON response, so you can confirm the response shape
(assumed to be {"rankings": [{"index": int, "logit": float}, ...]}) actually
matches what api_reranker.py expects, rather than trusting it blind.

Usage:
    python test_reranker_sanity.py

Requires RERANKER_API_KEY in your .env (same key you already added).
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY_ENV_VAR = "RERANKER_API_KEY"
MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"
URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking"

QUERY = "What is the capital of France?"
PASSAGES = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany.",
    "The Eiffel Tower is located in Paris, France.",
    "Bananas are a good source of potassium.",
]


def main():
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV_VAR} not found -- check your .env file")

    payload = {
        "model": MODEL,
        "query": {"text": QUERY},
        "passages": [{"text": p} for p in PASSAGES],
        "truncate": "END",
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    print(f"POST {URL}")
    print(f"Query: {QUERY!r}")
    print(f"Passages: {len(PASSAGES)}")
    print("-" * 60)

    response = requests.post(URL, headers=headers, json=payload, timeout=30)

    print(f"Status: {response.status_code}")
    print("-" * 60)

    try:
        data = response.json()
        print(json.dumps(data, indent=2))
    except ValueError:
        print("Response was not JSON:")
        print(response.text)
        return

    response.raise_for_status()

    rankings = data.get("rankings")
    if rankings is None:
        print("\nWARNING: no 'rankings' key found in the response -- api_reranker.py")
        print("expects this key. Check the actual key name printed above and update")
        print("APIReranker.rerank() to match it.")
        return

    print(f"\nOK: found {len(rankings)} rankings.")
    print("Expected top result: the Paris-related passages (index 0 or 2) should")
    print("rank above Berlin or bananas. If they don't, something's off with the")
    print("request shape, not just the code.\n")

    for entry in rankings:
        idx = entry.get("index")
        logit = entry.get("logit")
        preview = PASSAGES[idx][:60] if idx is not None and idx < len(PASSAGES) else "???"
        print(f"  index={idx} logit={logit}  -> {preview!r}")


if __name__ == "__main__":
    main()