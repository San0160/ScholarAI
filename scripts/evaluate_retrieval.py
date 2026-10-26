"""
Retrieval evaluation for ScholarAI.

Scores every question set twice: with the configured reranker, and without it
(vector search only). The difference between the two runs is what the reranker
adds.

Run from the project root (the folder that contains config/ and src/):

    python scripts/evaluate_retrieval.py

With no options it uses the PDFs in data/documents and the three question files
in src/ai_research_assistant/evaluation/. Useful options:

    --k 5                       score the top 5 instead of the top 3
    --dump-chunks FILE.md       write every chunk (file, page, id, opening words)
                                to FILE.md so you can check your page labels

How the test indexes are built
------------------------------
The web app holds ONE document at a time. To test the same situation, each
document named in a "document" field gets its own small index, and questions
about that document are searched only there. Questions without a "document"
field, and the multi-document file, are searched in one index that holds every
PDF. All of these indexes live under artifacts/eval_index/, so your app's index
in data/vector_db is never touched and the web app can keep running.

Needs in .env: NVIDIA_API_KEY (embeddings) and RERANKER_API_KEY (reranker).
Does not call the LLM, so it does not use your Groq quota.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.evaluation.document_retrieval_evaluator import DocumentRetrievalEvaluator
from ai_research_assistant.evaluation.eval_indexes import build_index
from ai_research_assistant.evaluation.retrieval_evaluator import RetrievalEvaluator
from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.reranking.no_reranker import NoReranker

EVAL_DIR = "src/ai_research_assistant/evaluation"
DEFAULT_QUESTIONS = [f"{EVAL_DIR}/questions.json", f"{EVAL_DIR}/question_attention.json"]
DEFAULT_MULTI_DOC = [f"{EVAL_DIR}/question_multi_doc.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ScholarAI retrieval with and without the reranker.")
    parser.add_argument("--docs", default="data/documents", help="Folder containing the documents the questions refer to.")
    parser.add_argument("--extensions", nargs="*", default=[".pdf"],
                        help="File types to index from --docs (default: .pdf only, so copies like attention.txt are ignored).")
    parser.add_argument("--questions", nargs="*", default=DEFAULT_QUESTIONS,
                        help="Question files labelled with evidence / expected_pages / expected_chunks.")
    parser.add_argument("--multi-doc", nargs="*", default=DEFAULT_MULTI_DOC,
                        help="Question files labelled with expected_source (which file should be found).")
    parser.add_argument("--k", type=int, default=3, help="Number of results scored per question.")
    parser.add_argument("--output-dir", default="artifacts/eval", help="Where the JSON report is written.")
    parser.add_argument("--dump-chunks", help="Write every indexed chunk to this Markdown file and exit.")
    return parser.parse_args()


class IndexRouter:
    """Looks like a RetrievalPipeline to the evaluators, but sends each query to
    the right index: the document's own index when the query is scoped to one
    file, otherwise the index that holds every document."""

    def __init__(self, combined: RetrievalPipeline, per_document: dict[str, RetrievalPipeline]):
        self.combined = combined
        self.per_document = per_document
        self._pipelines = [combined, *per_document.values()]
        self._configured_rerankers = [pipeline.reranker for pipeline in self._pipelines]

    def use_reranker(self, enabled: bool) -> None:
        for pipeline, configured in zip(self._pipelines, self._configured_rerankers):
            pipeline.reranker = configured if enabled else NoReranker()

    def run(self, query, metadata_filters=None, min_score=None, top_k=None):
        pipeline = self.combined
        if metadata_filters and set(metadata_filters) == {"filename"}:
            pipeline = self.per_document.get(metadata_filters["filename"], self.combined)
        return pipeline.run(query, metadata_filters=metadata_filters, min_score=min_score, top_k=top_k)


def safe(getter, default=None):
    try:
        return getter()
    except Exception:
        return default


def config_snapshot(config, k: int) -> dict:
    """The settings that decide retrieval quality, saved with every report so
    results from different runs can be compared fairly."""
    commit = safe(lambda: subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip())
    return {
        "k": k,
        "git_commit": commit,
        "embedding_model": safe(lambda: config.embeddings.model),
        "chunk_size": safe(lambda: config.chunking.chunk_size),
        "chunk_overlap": safe(lambda: config.chunking.chunk_overlap),
        "candidate_k": safe(lambda: config.retrieval.candidate_k),
        "reranker_provider": safe(lambda: config.reranking.provider),
        "reranker_model": safe(lambda: config.reranking.model),
    }


def load_json(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def check_labels(name: str, questions: list[dict], indexed_files: set, indexed_chunks: set) -> None:
    """Warn about labels that cannot match anything in the current index."""
    for item in questions:
        document = item.get("document") or item.get("expected_source")
        if document and document not in indexed_files:
            print(f"  WARNING [{name}] '{document}' is not in --docs: {item['question']}")
        stale = [chunk for chunk in item.get("expected_chunks", []) if chunk not in indexed_chunks]
        if stale and not item.get("evidence") and not item.get("expected_pages"):
            print(f"  WARNING [{name}] chunk ids not in the current index {stale}: {item['question']}")


def dump_chunks(chunks, path: str) -> None:
    lines = ["| file | page | chunk_id | opening words |", "| --- | --- | --- | --- |"]
    for chunk in chunks:
        metadata = chunk.metadata
        opening = " ".join(chunk.page_content.split())[:120].replace("|", "/")
        lines.append(f"| {metadata.get('filename')} | {metadata.get('page', '')} | {metadata.get('chunk_id')} | {opening} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks to {path}")


def main() -> int:
    args = parse_args()
    config = ConfigurationManager().config

    extensions = {extension.lower() for extension in args.extensions}
    files = sorted(path for path in Path(args.docs).iterdir() if path.suffix.lower() in extensions)
    if not files:
        print(f"No {sorted(extensions)} files found in {args.docs}")
        return 1
    files_by_name = {path.name: path for path in files}

    print(f"Building test indexes from {len(files)} document(s) in {args.docs} ...")
    combined = build_index("all_documents", files)
    chunks = combined.vector_store.get_documents()
    indexed_files = {chunk.metadata.get("filename") for chunk in chunks}
    indexed_chunks = {chunk.metadata.get("chunk_id") for chunk in chunks}
    print(f"  combined index: {len(chunks)} chunks from {len(indexed_files)} file(s).")

    if args.dump_chunks:
        dump_chunks(chunks, args.dump_chunks)
        return 0

    question_sets = [(path, load_json(path), "chunk") for path in args.questions]
    question_sets += [(path, load_json(path), "document") for path in args.multi_doc]
    if not question_sets:
        print("No question files given (--questions / --multi-doc).")
        return 1

    for path, questions, _ in question_sets:
        check_labels(Path(path).name, questions, indexed_files, indexed_chunks)

    # One small index per document that a chunk-level question is scoped to.
    scoped_documents = sorted({
        item["document"]
        for _, questions, level in question_sets if level == "chunk"
        for item in questions
        if item.get("document") in files_by_name
    })
    per_document = {name: build_index(Path(name).stem, [files_by_name[name]]) for name in scoped_documents}

    router = IndexRouter(combined, per_document)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config_snapshot(config, args.k),
        "documents": [path.name for path in files],
        "results": {},
    }

    for configuration, with_reranker in (("with_reranker", True), ("without_reranker", False)):
        router.use_reranker(with_reranker)
        print(f"Scoring questions ({configuration}) ...")
        for path, questions, level in question_sets:
            if level == "chunk":
                result = RetrievalEvaluator(router, k=args.k).evaluate(questions)
            else:
                result = DocumentRetrievalEvaluator(router, k=args.k).evaluate(questions)
            report["results"].setdefault(Path(path).name, {})[configuration] = result

    print_table(report, args.k)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"retrieval_{stamp}.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull report with per-question details: {output_path}")
    return 0


def print_table(report: dict, k: int) -> None:
    print()
    print(f"{'question set':<32} {'reranker':<10} {'hits':>8} {f'hit@{k}':>7} {f'recall@{k}':>10} {'MRR':>6} {'ms':>7}")
    print("-" * 86)
    for set_name, by_configuration in report["results"].items():
        for configuration, result in by_configuration.items():
            hits = result.get("hits", round(result.get("hit_rate", 0.0) * len(result.get("details", []))))
            total = result.get("questions", len(result.get("details", [])))
            recall = result.get("recall")
            recall_text = f"{recall:.2f}" if recall is not None else "-"
            latency = result.get("mean_latency_ms")
            latency_text = f"{latency:.0f}" if latency is not None else "-"
            print(
                f"{set_name[:32]:<32} {'yes' if configuration == 'with_reranker' else 'no':<10} "
                f"{f'{hits}/{total}':>8} {result['hit_rate']:>7.2f} {recall_text:>10} {result['mrr']:>6.2f} {latency_text:>7}"
            )
        skipped = next(iter(by_configuration.values())).get("skipped")
        if skipped:
            print(f"  skipped (no gold labels): {len(skipped)}")


if __name__ == "__main__":
    sys.exit(main())
