import re
from pathlib import Path

from ai_research_assistant.pipeline.indexing_pipeline import IndexingPipeline
from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline

# Evaluation indexes live here, so the web app's own index is never touched.
INDEX_ROOT = Path("artifacts/eval_index")


def build_index(name: str, files: list[Path]) -> RetrievalPipeline:
    """Index `files` into a fresh store under artifacts/eval_index/<name> and
    return a retrieval pipeline that searches it."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    storage_path = str(INDEX_ROOT / safe_name)
    print(f"  indexing {len(files)} file(s) -> {storage_path}")
    IndexingPipeline(storage_path=storage_path).run([str(path) for path in files], rebuild=True)
    return RetrievalPipeline(storage_path=storage_path)
