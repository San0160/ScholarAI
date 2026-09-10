from ai_research_assistant.chunking.base_chunker import BaseChunker
from ai_research_assistant.chunking.recursive_chunker import RecursiveChunker
from ai_research_assistant.entity.config_entity import ChunkingConfig
from ai_research_assistant.utils.tokenizer_utils import get_token_counter
from ai_research_assistant.utils.text_segmentation import get_sentence_splitter


def create_chunker(config: ChunkingConfig) -> BaseChunker:
    token_counter = get_token_counter(config.embeddings.model)
    sentence_splitter = get_sentence_splitter()

    return RecursiveChunker(
        chunk_size = config.chunking.chunk_size,
        chunk_overlap = config.chunking.chunk_overlap,
        token_counter = token_counter,
        sentence_splitter=sentence_splitter,
    )