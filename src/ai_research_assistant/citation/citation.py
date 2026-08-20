from dataclasses import dataclass


@dataclass
class Citation:

    source_id: int
    document: str
    page: int | str
    chunk_id: str