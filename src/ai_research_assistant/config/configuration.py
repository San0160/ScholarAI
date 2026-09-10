from ai_research_assistant.utils.common import *
from ai_research_assistant.constants import *
from ai_research_assistant.entity.config_entity import ChunkingConfig, EmbeddingConfig

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH):     # Access to constants
        self.config = read_yaml(config_filepath) # read all config and params yaml files
        create_directories([self.config.artifacts_root])

    def get_chunking_config(self) -> ChunkingConfig:
        return ChunkingConfig(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            embedding_model=self.config.embeddings.model,
        )

    def get_embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            provider=self.config.embeddings.provider,
            model=self.config.embeddings.model,
            device=self.config.embeddings.device,
            batch_size=self.config.embeddings.batch_size,
            show_progress_bar=self.config.embeddings.show_progress_bar,
        )