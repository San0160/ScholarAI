"""LLM-as-judge grading of generated answers.

The judge scores two things on a 1 to 5 scale:

  faithfulness  is every claim in the answer supported by the excerpts the
                assistant was given? (catches made-up details)
  correctness   does the answer agree with a reference answer written by a
                person? (only when a reference is available)

Use a judge from a different model family than the one that writes the
answers. A model tends to rate its own writing too kindly.
"""

import json
import logging
import re

from ai_research_assistant.evaluation.answer_checks import strip_markers

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a strict grader of answers produced by a question-answering assistant.
You are given a QUESTION, the CONTEXT excerpts the assistant was allowed to use, the ASSISTANT ANSWER, and sometimes a REFERENCE ANSWER written by a person.

Judge only from the texts you are given. Do not use outside knowledge. Do not reward length or style.
Ignore citation markers such as [Source 2] in the answer.

FAITHFULNESS (1 to 5): is every statement in the ASSISTANT ANSWER supported by the CONTEXT?
5 = every statement is supported
4 = supported, with only trivial wording added
3 = one statement is not in the context or is stretched beyond it
2 = several statements are not supported
1 = mostly unsupported, or contradicts the context
An answer that only says the information is not available makes no claims: score it 5.

CORRECTNESS (1 to 5): does the ASSISTANT ANSWER agree with the REFERENCE ANSWER?
5 = every key point of the reference is present and nothing contradicts it
4 = one minor point is missing
3 = about half of the key points are present
2 = most key points are missing or vague
1 = wrong, contradicts the reference, or says the information is not available
If no REFERENCE ANSWER is given, set correctness to null.

Reply with one JSON object and nothing else:
{"faithfulness": <1-5>, "faithfulness_reason": "<one sentence>", "correctness": <1-5 or null>, "correctness_reason": "<one sentence>"}"""


class LLMJudge:
    """Grades answers with any object that has generate(messages) -> str."""

    def __init__(self, llm):
        self.llm = llm

    def grade(self, question: str, answer: str, context: str, reference: str | None = None) -> dict:
        user_message = (
            f"QUESTION:\n{question}\n\n"
            f"REFERENCE ANSWER:\n{reference if reference else 'none provided'}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ASSISTANT ANSWER:\n{strip_markers(answer).strip()}"
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        parsed = None
        for attempt in range(2):
            reply = self.llm.generate(messages)
            parsed = parse_judge_reply(reply, expect_correctness=bool(reference))
            if parsed is not None:
                break
            logger.warning("Judge reply was not valid JSON (attempt %d): %r", attempt + 1, reply[:200])
            messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "Reply with the JSON object only."},
            ]

        if parsed is None:
            return {
                "faithfulness": None, "faithfulness_reason": None,
                "correctness": None, "correctness_reason": None,
                "judge_error": "judge did not return valid JSON",
            }

        parsed["judge_error"] = None
        return parsed


def _score(value):
    """1 to 5 as an int, or None."""
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def parse_judge_reply(reply: str, expect_correctness: bool) -> dict | None:
    """Pull the JSON object out of a judge reply. None if it can't be used."""
    match = re.search(r"\{.*\}", reply or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    faithfulness = _score(data.get("faithfulness"))
    correctness = _score(data.get("correctness")) if expect_correctness else None

    if faithfulness is None or (expect_correctness and correctness is None):
        return None

    return {
        "faithfulness": faithfulness,
        "faithfulness_reason": str(data.get("faithfulness_reason") or "").strip(),
        "correctness": correctness,
        "correctness_reason": str(data.get("correctness_reason") or "").strip() if expect_correctness else "",
    }
