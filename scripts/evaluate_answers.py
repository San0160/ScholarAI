"""
Answer evaluation for ScholarAI (phase 2).

Asks every labelled question through the same code the web app uses, then
checks the answer in four ways:

  1. Citations   did the model cite a page that really holds the answer?
  2. Key facts   does the answer contain the facts listed in "must_include"?
  3. Faithfulness (LLM judge, 1-5)  is every claim supported by the excerpts?
  4. Correctness  (LLM judge, 1-5)  does it match the "expected_answer"?

Checks 1 and 2 need no LLM. Checks 3 and 4 use the judge model set under
evaluation.judge in config/config.yaml (a different model family from the one
that writes the answers).

Run from the project root:

    python scripts/evaluate_answers.py                 # everything
    python scripts/evaluate_answers.py --limit 3       # try the first 3 questions
    python scripts/evaluate_answers.py --no-judge      # citations and key facts only

Needs in .env: GROQ_API_KEY (answers), NVIDIA_API_KEY (embeddings and judge)
and RERANKER_API_KEY. This script DOES use your Groq quota. On the free plan
(8K tokens per minute) it pauses between questions and waits out rate limits,
so 26 questions take roughly 15 to 25 minutes. A partial report is saved after
every question, so an interrupted run still leaves results.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.evaluation.answer_checks import average, citation_check, fact_check
from ai_research_assistant.evaluation.eval_indexes import build_index
from ai_research_assistant.evaluation.llm_judge import LLMJudge
from ai_research_assistant.llm.api_llm import APILLM
from ai_research_assistant.pipeline.answer_service import answer_question
from ai_research_assistant.pipeline.generation_pipeline import GenerationPipeline

EVAL_DIR = "src/ai_research_assistant/evaluation"
DEFAULT_QUESTIONS = [f"{EVAL_DIR}/question_attention.json", f"{EVAL_DIR}/questions.json"]
MAX_CONSECUTIVE_ERRORS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ScholarAI answers: citations, key facts and an LLM judge.")
    parser.add_argument("--docs", default="data/documents", help="Folder containing the documents the questions refer to.")
    parser.add_argument("--questions", nargs="*", default=DEFAULT_QUESTIONS, help="Question files to run.")
    parser.add_argument("--limit", type=int, help="Only run the first N questions (handy for a quick test).")
    parser.add_argument("--pause", type=float, default=20.0,
                        help="Seconds to wait between questions, to stay under the LLM rate limit.")
    parser.add_argument("--judge-pause", type=float, default=1.0, help="Seconds to wait after each judge call.")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge.")
    parser.add_argument("--output-dir", default="artifacts/eval", help="Where the JSON report is written.")
    return parser.parse_args()


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


def load_items(paths: list[str]) -> list[dict]:
    """Questions that have a document and at least one thing to check."""
    items = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            for item in json.load(file):
                has_label = item.get("expected_pages") or item.get("must_include") or item.get("expected_answer")
                if item.get("document") and has_label:
                    items.append({**item, "_set": Path(path).name})
    return items


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
        try:
            names = sorted(model.id for model in llm.client.models.list())
            likely = [name for name in names if any(tag in name.lower() for tag in ("instruct", "chat", "nemotron", "mistral", "llama"))]
            print("Chat models your key can list (pick one and set evaluation.judge.model in config.yaml):")
            for name in likely[:40]:
                print(f"  {name}")
        except Exception:
            print("Could not list models either. Check NVIDIA_API_KEY in .env and the model name in config.yaml.")
        raise SystemExit(2)
    return LLMJudge(llm)


def evaluate_one(item, retrieval, generation, judge, judge_pause) -> dict:
    document = item["document"]
    row = {
        "set": item["_set"],
        "document": document,
        "question": item["question"],
        "expected_answer": item.get("expected_answer"),
        "error": None,
    }

    start = time.perf_counter()
    outcome = with_retries(
        lambda: answer_question(retrieval, generation, item["question"], metadata_filters={"filename": document}),
        label="answer",
    )
    row["answer_seconds"] = round(time.perf_counter() - start, 1)

    if outcome is None:
        row["error"] = "nothing was retrieved for this question"
        return row

    row["answer"] = outcome.answer
    row["model_cited"] = outcome.model_cited
    context_pages = [source.get("page") for source in outcome.context_sources]
    row["context_pages"] = sorted({page for page in context_pages if isinstance(page, int)})

    if item.get("expected_pages"):
        cited_pages = [source.get("page") for source in outcome.citations] if outcome.model_cited else []
        row["gold_pages"] = item["expected_pages"]
        row.update(citation_check(cited_pages, item["expected_pages"], context_pages))

    if item.get("must_include"):
        row.update(fact_check(outcome.answer, item["must_include"]))

    if judge is not None:
        grade = with_retries(
            lambda: judge.grade(item["question"], outcome.answer, outcome.context, item.get("expected_answer")),
            label="judge",
        )
        row.update(grade)
        time.sleep(judge_pause)

    return row


def summarize(rows: list[dict]) -> dict:
    ok = [row for row in rows if not row.get("error")]
    with_citation = [row for row in ok if "citation_hit" in row]
    return {
        "questions": len(rows),
        "answered": len(ok),
        "model_cited": average(with_citation, "model_cited"),
        "citation_hit": average(with_citation, "citation_hit"),
        "citation_precision": average(with_citation, "citation_precision"),
        "gold_in_context": average(with_citation, "gold_in_context"),
        "fact_recall": average(ok, "fact_recall"),
        "faithfulness": average(ok, "faithfulness"),
        "correctness": average(ok, "correctness"),
        "answer_seconds": average(ok, "answer_seconds"),
    }


def fmt(value, digits=2, percent=False) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.0f}%" if percent else f"{value:.{digits}f}"


def print_summary(rows: list[dict]) -> None:
    sets = sorted({row["set"] for row in rows})
    groups = [(name, [row for row in rows if row["set"] == name]) for name in sets]
    if len(groups) > 1:
        groups.append(("ALL", rows))

    print()
    print(f"{'question set':<26} {'n':>3} {'cited':>6} {'page ok':>8} {'cite prec':>9} {'in prompt':>9} {'facts':>6} {'faithful':>8} {'correct':>8} {'sec':>5}")
    print("-" * 100)
    for name, group in groups:
        s = summarize(group)
        print(
            f"{name[:26]:<26} {s['answered']:>3} {fmt(s['model_cited'], percent=True):>6} "
            f"{fmt(s['citation_hit'], percent=True):>8} {fmt(s['citation_precision'], percent=True):>9} "
            f"{fmt(s['gold_in_context'], percent=True):>9} {fmt(s['fact_recall'], percent=True):>6} "
            f"{fmt(s['faithfulness']):>8} {fmt(s['correctness']):>8} {fmt(s['answer_seconds'], 1):>5}"
        )
    print()
    print("cited = the model cited a source itself | page ok = a cited page holds the answer")
    print("cite prec = share of cited pages that hold the answer | in prompt = a right page reached the LLM")
    print("facts = key facts found | faithful / correct = judge scores out of 5")


def config_snapshot(config_manager, judge_enabled: bool) -> dict:
    config = config_manager.config
    commit = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        pass
    return {
        "git_commit": commit,
        "answer_model": config.llm.model,
        "judge_model": config_manager.get_judge_config().model if judge_enabled else None,
        "embedding_model": config.embeddings.model,
        "reranker_model": config.reranking.model,
        "candidate_k": config.retrieval.candidate_k,
        "rerank_top_k": config.reranking.top_k,
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
        "max_context_tokens": config.llm.max_context_tokens,
    }


def save_report(path: Path, snapshot: dict, rows: list[dict]) -> None:
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": snapshot,
        "summary": summarize(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_manager = ConfigurationManager()

    items = load_items(args.questions)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("No questions with a document and labels found.")
        return 1

    documents = sorted({item["document"] for item in items})
    missing = [name for name in documents if not (Path(args.docs) / name).exists()]
    if missing:
        print(f"These documents are not in {args.docs}: {missing}")
        return 1

    judge = None if args.no_judge else make_judge(config_manager)

    print(f"Building test indexes for {len(documents)} document(s) ...")
    retrievals = {name: build_index(Path(name).stem, [Path(args.docs) / name]) for name in documents}

    embedder = EmbeddingFactory.create_embedding(config_manager.get_embedding_config())
    generation = GenerationPipeline(embedder=embedder)

    snapshot = config_snapshot(config_manager, judge is not None)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    rows = []
    consecutive_errors = 0
    print(f"\nAnswering {len(items)} question(s). A partial report is saved after each one.\n")

    for number, item in enumerate(items, start=1):
        label = f"[{number}/{len(items)}] {ascii_text(item['question'])[:70]}"
        try:
            row = evaluate_one(item, retrievals[item["document"]], generation, judge, args.judge_pause)
        except Exception as error:
            cause = error.__cause__ or error
            row = {"set": item["_set"], "document": item["document"], "question": item["question"],
                   "error": f"{type(cause).__name__}: {cause}"}

        rows.append(row)
        save_report(output_path, snapshot, rows)

        if row.get("error"):
            consecutive_errors += 1
            print(f"{label}\n    ERROR: {ascii_text(row['error'])[:200]}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"\nStopping after {MAX_CONSECUTIVE_ERRORS} errors in a row. Fix the problem above and run again.")
                break
        else:
            consecutive_errors = 0
            parts = []
            if "citation_hit" in row:
                parts.append("page " + ("ok" if row["citation_hit"] else "WRONG" if row["model_cited"] else "NOT CITED"))
            if row.get("fact_recall") is not None:
                parts.append(f"facts {row['fact_recall'] * 100:.0f}%")
            if row.get("faithfulness") is not None:
                parts.append(f"faithful {row['faithfulness']}")
            if row.get("correctness") is not None:
                parts.append(f"correct {row['correctness']}")
            if row.get("judge_error"):
                parts.append("judge failed")
            print(f"{label}\n    " + " | ".join(parts))

        if number < len(items):
            time.sleep(args.pause)

    print_summary(rows)
    print(f"\nFull report with every answer and judge comment: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
