from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.vector_store.vector_store_factory import VectorStoreFactory


config_manager = ConfigurationManager()

embedder = EmbeddingFactory.create_embedding(config_manager.get_embedding_config())

vector_store = VectorStoreFactory.create_vector_store(
    dimension=embedder.dimension
)

vector_store.load()

print(
    "\nTotal vectors:",
    vector_store.index.ntotal
)

print("\n" + "=" * 100)

for index, document in enumerate(
    vector_store.documents
):

    content = document.page_content.replace(
        "\n",
        " "
    )

    print(
        f"{index} | "
        f"{document.metadata.get('chunk_id')} | "
        f"Page {document.metadata.get('page')} | "
        f"{document.metadata.get('section')} | "
        f"{content[:250]}..."
    )