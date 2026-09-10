from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int
    embedding_model: str


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    device: str
    batch_size: int
    show_progress_bar: bool