"""
Check that the judge model can tell good answers from bad ones.

An LLM judge that gives every answer 5 out of 5 is useless. This script tests the
judge on answers whose quality we already know:

  good        the reference answer itself               -> should score high on both
  invented    the reference answer plus one false claim -> faithfulness should drop
  wrong       the answer to a different question        -> correctness should drop

    python scripts/check_judge.py              # 8 questions, 24 judge calls
    python scripts/check_judge.py --count 12

It does not use Groq. It only calls the judge model from evaluation.judge in config.yaml.
Run it whenever you change the judge model.
"""

import argparse
import json
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.evaluation.judge_setup import make_judge, with_retries

DEFAULT_QUESTIONS = "src/ai_research_assistant/evaluation/question_attention.json"

# One false claim added to a correct answer. None of these is in the paper.
INVENTED_CLAIMS = [
    "The model was trained for 12 days on 64 TPU v5 chips.",
    "The authors also reported that it beat every model on the GLUE benchmark.",
    "Training used the SGD optimizer with a fixed learning rate of 0.1.",
    "The paper states that the final model has 2.7 billion parameters.",
]
PASS_RATE_NEEDED = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that the LLM judge separates good answers from bad ones.")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--count", type=int, default=8, help="How many questions to test with.")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to wait between judge calls.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.questions, encoding="utf-8") as file:
        items = [item for item in json.load(file) if item.get("expected_answer")][: args.count]
    if len(items) < 2:
        print("Need at least two questions with an expected_answer.")
        return 1

    judge = make_judge(ConfigurationManager())
    print(f"Testing the judge on {len(items)} questions (3 answers each) ...\n")

    results = {"good": [], "invented": [], "wrong": []}

    def grade(item, answer):
        reply = with_retries(
            lambda: judge.grade(item["question"], answer, item["expected_answer"], item["expected_answer"]),
            label="judge",
        )
        time.sleep(args.pause)
        return reply

    for number, item in enumerate(items):
        other = items[(number + 1) % len(items)]
        results["good"].append(grade(item, item["expected_answer"]))
        results["invented"].append(grade(item, item["expected_answer"] + " " + INVENTED_CLAIMS[number % len(INVENTED_CLAIMS)]))
        results["wrong"].append(grade(item, other["expected_answer"]))
        print(f"  [{number + 1}/{len(items)}] done")

    def passed(case, grade_row):
        faithfulness, correctness = grade_row["faithfulness"], grade_row["correctness"]
        if faithfulness is None or correctness is None:
            return False
        if case == "good":
            return faithfulness >= 4 and correctness >= 4
        if case == "invented":
            return faithfulness <= 3
        return correctness <= 2

    rules = {
        "good": "faithful 4-5 and correct 4-5",
        "invented": "faithful 3 or lower",
        "wrong": "correct 2 or lower",
    }
    print(f"\n{'answer type':<12} {'avg faithful':>12} {'avg correct':>12} {'passed':>8}   rule")
    print("-" * 78)
    all_ok = True
    for case, rows in results.items():
        scored = [row for row in rows if row["faithfulness"] is not None]
        avg_f = sum(row["faithfulness"] for row in scored) / len(scored) if scored else float("nan")
        avg_c = sum(row["correctness"] for row in scored) / len(scored) if scored else float("nan")
        ok = sum(passed(case, row) for row in rows)
        rate = ok / len(rows)
        all_ok = all_ok and rate >= PASS_RATE_NEEDED
        print(f"{case:<12} {avg_f:>12.2f} {avg_c:>12.2f} {f'{ok}/{len(rows)}':>8}   {rules[case]}")

    failed_calls = sum(1 for rows in results.values() for row in rows if row.get("judge_error"))
    if failed_calls:
        print(f"\n{failed_calls} judge call(s) did not return usable JSON.")

    print()
    if all_ok:
        print("RESULT: this judge separates good, invented and wrong answers. It is usable.")
        return 0
    print("RESULT: this judge is too lenient or unreliable. Use a larger model (python scripts/find_judge.py)")
    print("        and run this check again before trusting any faithfulness or correctness score.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
