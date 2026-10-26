import re
import time
import unicodedata
from statistics import mean


def _normalize(text: str) -> str:
    """Lowercase, fold ligatures (e.g. "ﬁ" -> "fi") and collapse whitespace, so a
    quote copied from a PDF viewer still matches the cleaned chunk text."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


class RetrievalEvaluator:
    """Chunk-level retrieval evaluation that survives re-chunking.

    Each question names its gold evidence in one of three ways, most precise first:

      evidence         short quotes copied from the document (5 to 12 words each).
                       A retrieved chunk is relevant if it contains one of them.
      expected_pages   page numbers. A chunk is relevant if it comes from one of
                       those pages of the question's document.
      expected_chunks  chunk ids, kept so older label files still load. These go
                       stale whenever chunking changes, so prefer the other two.

    With a "document" field, the search is restricted to that file, the same way
    the web app scopes questions to the uploaded document.

    Metrics, over the top k results:
      hit_rate  share of questions with at least one relevant chunk
      recall    share of the gold quotes / pages / chunks that were found
      mrr       mean of 1 / rank of the first relevant chunk (0 if none)

    Precision is deliberately left out: with one or two gold units per question
    and k = 3, the best possible precision is 0.33 to 0.67, which says more
    about the label set than about retrieval.
    """

    def __init__(self, pipeline, k: int = 3, scope_to_document: bool = True):
        self.pipeline = pipeline
        self.k = k
        self.scope_to_document = scope_to_document

    def evaluate(self, questions: list[dict]) -> dict:
        details = []
        skipped = []

        for item in questions:
            gold_type, gold = self._gold(item)
            if not gold:
                skipped.append(item.get("question", "<missing question>"))
                continue

            document = item.get("document")
            filters = {"filename": document} if (self.scope_to_document and document) else None

            start = time.perf_counter()
            results = self.pipeline.run(item["question"], metadata_filters=filters, top_k=self.k)
            latency_ms = (time.perf_counter() - start) * 1000
            results = results[: self.k]

            found = set()
            first_relevant_rank = None
            retrieved = []

            for rank, result in enumerate(results, start=1):
                matched = self._match(result.document, gold_type, gold, document)
                found.update(matched)
                if matched and first_relevant_rank is None:
                    first_relevant_rank = rank

                metadata = result.document.metadata
                retrieved.append({
                    "rank": rank,
                    "chunk_id": metadata.get("chunk_id"),
                    "filename": metadata.get("filename"),
                    "page": metadata.get("page"),
                    "score": round(float(result.score), 4),
                    "relevant": bool(matched),
                })

            details.append({
                "question": item["question"],
                "document": document,
                "gold_type": gold_type,
                "gold": sorted(gold, key=str),
                "hit": first_relevant_rank is not None,
                "first_relevant_rank": first_relevant_rank,
                "reciprocal_rank": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
                "recall": len(found) / len(gold),
                "latency_ms": round(latency_ms, 1),
                "retrieved": retrieved,
            })

        return self._summarize(details, skipped)

    @staticmethod
    def _gold(item: dict) -> tuple[str | None, set]:
        if item.get("evidence"):
            return "evidence", {_normalize(quote) for quote in item["evidence"] if quote.strip()}
        if item.get("expected_pages"):
            return "pages", {int(page) for page in item["expected_pages"]}
        if item.get("expected_chunks"):
            return "chunks", set(item["expected_chunks"])
        return None, set()

    @staticmethod
    def _match(document, gold_type: str, gold: set, expected_document: str | None) -> set:
        """Return the gold units this retrieved chunk satisfies."""
        metadata = document.metadata

        if expected_document and metadata.get("filename") != expected_document:
            return set()

        if gold_type == "evidence":
            text = _normalize(document.page_content)
            return {quote for quote in gold if quote in text}

        if gold_type == "pages":
            return {metadata.get("page")} & gold

        if gold_type == "chunks":
            return {metadata.get("chunk_id")} & gold

        return set()

    def _summarize(self, details: list[dict], skipped: list[str]) -> dict:
        if not details:
            return {
                "k": self.k, "questions": 0, "hits": 0, "hit_rate": 0.0,
                "recall": 0.0, "mrr": 0.0, "mean_latency_ms": 0.0,
                "skipped": skipped, "details": [],
            }

        hits = sum(detail["hit"] for detail in details)
        return {
            "k": self.k,
            "questions": len(details),
            "hits": hits,
            "hit_rate": hits / len(details),
            "recall": mean(detail["recall"] for detail in details),
            "mrr": mean(detail["reciprocal_rank"] for detail in details),
            "mean_latency_ms": round(mean(detail["latency_ms"] for detail in details), 1),
            "skipped": skipped,
            "details": details,
        }
