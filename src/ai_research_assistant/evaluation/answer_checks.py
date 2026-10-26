"""Checks on a generated answer that need no LLM: key facts and cited pages."""

import re
import unicodedata
from statistics import mean

_DASHES = re.compile("[‐-―−]")

# [Source 3], (Source 3), and the document's own reference markers like [20] or [3, 4].
_MARKERS = re.compile(
    r"\[\s*source[^\]]*\]|\(\s*source[^)]*\)|\[\s*\d+(?:\s*[,;–-]\s*\d+)*\s*\]",
    re.IGNORECASE,
)


def strip_markers(text: str) -> str:
    """Remove citation markers, so a number inside "[Source 6]" is never
    mistaken for a number in the answer itself."""
    return _MARKERS.sub(" ", text or "")


def normalize(text: str) -> str:
    """Lowercase, fold ligatures and dash variants, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _DASHES.sub("-", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _term_pattern(term: str) -> re.Pattern:
    """A term matches as a whole word. A trailing * makes it a prefix, so
    "parallel*" also matches "parallelization" and "image*" matches "images"."""
    term = normalize(term)
    prefix = term.endswith("*")
    body = re.escape(term[:-1] if prefix else term)
    # A decimal point belongs to the number: "6" must not match inside "0.6" or "6.5".
    left = r"(?<![0-9a-z])(?<!\d\.)"
    right = "" if prefix else r"(?![0-9a-z])(?!\.\d)"
    return re.compile(left + body + right)


def fact_check(answer: str, must_include: list) -> dict:
    """Check that an answer contains every key fact.

    must_include is a list of facts. Each fact is a word or phrase, or a list of
    alternatives where any one is enough, e.g. [["six", "6"], "encoder"].
    Returns the share of facts found and the ones that were missing.
    """
    text = normalize(strip_markers(answer))
    missing = []

    for fact in must_include:
        alternatives = [fact] if isinstance(fact, str) else list(fact)
        if not any(_term_pattern(term).search(text) for term in alternatives):
            missing.append(alternatives[0] if len(alternatives) == 1 else alternatives)

    total = len(must_include)
    return {
        "fact_recall": (total - len(missing)) / total if total else None,
        "missing_facts": missing,
    }


def citation_check(cited_pages: list, gold_pages: list, context_pages: list) -> dict:
    """Compare the pages an answer cites with the pages that hold the answer.

    citation_hit        at least one cited page is a gold page
    citation_precision  share of cited pages that are gold pages
    gold_in_context     a gold page was in the prompt at all (a retrieval check,
                        so a wrong citation can be told apart from missing context)
    """
    gold = {int(page) for page in gold_pages}
    cited = {page for page in cited_pages if isinstance(page, int)}
    in_context = {page for page in context_pages if isinstance(page, int)}

    return {
        "cited_pages": sorted(cited),
        "citation_hit": bool(cited & gold),
        "citation_precision": len(cited & gold) / len(cited) if cited else 0.0,
        "gold_in_context": bool(in_context & gold),
    }


def average(rows: list[dict], key: str):
    """Mean of a metric over the rows that have it (True/False count as 1/0)."""
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return mean(values) if values else None
