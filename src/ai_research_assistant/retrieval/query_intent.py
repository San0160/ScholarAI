import re

_WHOLE_DOCUMENT_PATTERN = re.compile(
    r"\b(summar(y|ies|ise|ize|ised|ized|ising|izing)|overview|outline|tl;?dr|gist|"
    r"(key|main) (points|ideas|topics|takeaways|themes)|"
    r"(what|tell me)\b.{0,20}\b(document|pdf|paper|file|article|report)\b.{0,10}\babout|"
    r"tell me about (this|the) (document|pdf|paper|file|article|report))",
    re.IGNORECASE,
)


def is_whole_document_question(question: str) -> bool:
    """True for questions about the document as a whole (summaries, overviews),
    which similarity search answers badly."""
    return bool(_WHOLE_DOCUMENT_PATTERN.search(question))