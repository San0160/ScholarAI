from ai_research_assistant.utils.common import *
from ai_research_assistant.constants import *
from ai_research_assistant.entity.config_entity import ChunkingConfig, EmbeddingConfig
from ai_research_assistant.entity.config_entity import ChunkingConfig, EmbeddingConfig, LLMConfig, RerankingConfig

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH):     # Access to constants
        self.config = read_yaml(config_filepath)                # read all config and params yaml files
        create_directories([self.config.artifacts_root])

    def get_chunking_config(self) -> ChunkingConfig:
        return ChunkingConfig(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            embedding_model=self.config.embeddings.model,
        )

    def get_embedding_config(self) -> EmbeddingConfig:
        embeddings = self.config.embeddings
        return EmbeddingConfig(
            provider=embeddings.provider,
            model=embeddings.model,
            device=embeddings.get("device", "auto"),
            batch_size=embeddings.get("batch_size", 32),
            show_progress_bar=embeddings.get("show_progress_bar", False),
            api_base_url=embeddings.get("api_base_url"),
            api_key_env_var=embeddings.get("api_key_env_var"),
            dimension=embeddings.get("dimension"),
        )

    def get_llm_config(self) -> LLMConfig:
        llm = self.config.llm
        return LLMConfig(
            provider=llm.provider,
            model=llm.model,
            device=llm.get("device", "auto"),
            max_context_tokens=llm.max_context_tokens,
            max_new_tokens=llm.max_new_tokens,
            api_base_url=llm.get("api_base_url"),
            api_key_env_var=llm.get("api_key_env_var"),
        )

    def get_reranking_config(self) -> RerankingConfig:
        reranking = self.config.reranking
        return RerankingConfig(
            provider=reranking.provider,
            model=reranking.model,
            top_k=reranking.top_k,
            api_base_url=reranking.get("api_base_url"),
            api_key_env_var=reranking.get("api_key_env_var"),
        )