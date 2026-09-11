import logging
from pathlib import Path

from ai_research_assistant.chunking.chunker_factory import create_chunker
from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.ingestion.document_loader import DocumentLoader
from ai_research_assistant.utils.text_cleaner import TextCleaner
from ai_research_assistant.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class IndexingPipeline:
    """Runs the full ingestion -> clean -> chunk -> embed -> index pipeline
    and persists the result to the vector store.

    On construction, loads whatever vector store already exists on disk (if
    any), so by default this pipeline is INCREMENTAL: calling run() again
    later only processes and appends files it hasn't seen before, rather
    than reprocessing everything from scratch. Pass rebuild=True to a run()
    call to instead wipe the existing store and reprocess the given files
    from a clean slate.

    Limitation: "already seen" is tracked by filename only, not content --
    if a source file's content changes but its filename doesn't, a normal
    (non-rebuild) run() will skip it as already indexed rather than picking
    up the change. Use rebuild=True to force a file to be reprocessed.
    """

    def __init__(self, storage_path: str = None):
        
        """Builds the pipeline's components from config and loads any
        existing vector store at `storage_path` (or the configured default
        path if not given). Starts from an empty store if none exists yet.
        """

        config_manager = ConfigurationManager()

        self.loader = DocumentLoader()
        self.cleaner = TextCleaner()

        self.chunker = create_chunker(config_manager.get_chunking_config())

        self.embedder = EmbeddingFactory.create_embedding(
            config_manager.get_embedding_config()
        )

        self.vector_store = VectorStoreFactory.create_vector_store(
            dimension=self.embedder.dimension,
            storage_path=storage_path
        )

        try:
            self.vector_store.load()
            logger.info(
                "Loaded existing vector store with %d vector(s)",
                self.vector_store.index.ntotal,
            )
        except FileNotFoundError:
            logger.info("No existing vector store found -- starting fresh")

    def _indexed_filenames(self) -> set[str]:

        """Returns the set of filenames already present in the vector
        store's documents, used to skip already-indexed files on a
        non-rebuild run().
        """
        return {
            document.metadata.get("filename")
            for document in self.vector_store.documents
            if document.metadata.get("filename")
        }

    def run(self, file_paths: list[str], rebuild: bool = False) -> dict:
        """Indexes the given files.

        Args:
            file_paths: Paths to the source files to load, chunk, embed,
                and add to the vector store.
            rebuild: If True, clears the existing vector store first and
                reprocesses every path given, ignoring what was previously
                indexed. If False (default), any file whose filename is
                already in the store is skipped, and only new files are
                processed and appended -- this is the incremental-add mode.

        Returns:
            A dict with counts for this run: "files" and "documents" and
            "chunks" actually processed (0 for all three if every given
            file was skipped as already-indexed), and "vectors", the
            vector store's total count after this run.
        """

        if rebuild:
            logger.info("Rebuild requested -- clearing existing vector store")
            self.vector_store.clear()

        already_indexed = self._indexed_filenames() if not rebuild else set()

        paths_to_process = []
        skipped = []

        for file_path in file_paths:
            filename = Path(file_path).name

            if filename in already_indexed:
                skipped.append(filename)
                continue

            paths_to_process.append(file_path)

        if skipped:
            logger.warning(
                "Skipping %d file(s) already present in the vector store: %s "
                "(pass rebuild=True to reprocess everything)",
                len(skipped), skipped,
            )

        if not paths_to_process:
            logger.info("Nothing new to index")
            return {
                "files": 0,
                "documents": 0,
                "chunks": 0,
                "vectors": self.vector_store.index.ntotal,
            }

        logger.info("Starting indexing run for %d file(s)", len(paths_to_process))

        # 1. Load all documents
        documents = []

        for file_path in paths_to_process:
            loaded_documents = self.loader.load(file_path)
            documents.extend(loaded_documents)

        # 2. Clean text
        documents = self.cleaner.clean_documents(documents)

        # 3. Create chunks
        chunks = self.chunker.split_documents(documents)

        # 4. Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)

        # 5. Add to vector store
        self.vector_store.add_documents(chunks, embeddings)

        # 6. Persist index
        self.vector_store.save()

        logger.info(
            "Indexing run complete: %d file(s) -> %d document(s) -> %d chunk(s) "
            "-> %d vector(s) in store",
            len(paths_to_process), len(documents), len(chunks), self.vector_store.index.ntotal,
        )

        return {
            "files": len(paths_to_process),
            "documents": len(documents),
            "chunks": len(chunks),
            "vectors": self.vector_store.index.ntotal
        }

    def run_single(self, file_path: str, rebuild: bool = False) -> dict:

        """Convenience wrapper for indexing a single file. See run().
        """
        return self.run([file_path], rebuild=rebuild)