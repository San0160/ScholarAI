import re

# Matches a full "[Source ...]" or "[Sources ...]" bracket. Case-insensitive
# and tolerant of missing/extra whitespace, singular vs. plural, and multiple
# source ids in one bracket separated however the model phrases it
# ("1, 2", "1 and 2", "1, 2, and 3"). Anything after a colon inside the
# bracket is metadata (CitationFormatter's "filename, p. X"), not a source id.
SOURCE_BLOCK_PATTERN = re.compile(r"\[\s*Sources?[^\]]*\]", re.IGNORECASE)


def extract_source_ids(block: str) -> list[int]:
    """Given one bracketed match from SOURCE_BLOCK_PATTERN, return the source
    id(s) it names, ignoring anything after a colon (e.g. a page number,
    which is also a digit and must not be mistaken for a cited source id)."""
    inner = block.strip("[]")
    number_part = inner.split(":", 1)[0]
    return [int(n) for n in re.findall(r"\d+", number_part)]