from ai_research_assistant.pipeline.indexing_pipeline import (
    IndexingPipeline
)


# --------------------------------------------------
# Test Files
# --------------------------------------------------

FILE_PATHS = [
    "data/documents/attention.pdf",
    "data/documents/deeplab.pdf",
    "data/documents/Image_Segmentation.pdf",
    "data/documents/Text_Summariser.pdf",
]


# --------------------------------------------------
# Storage Path
# --------------------------------------------------

STORAGE_PATH = (
    "artifacts/vector_store/multi_document"
)


# --------------------------------------------------
# Index Documents
# --------------------------------------------------

indexing_pipeline = IndexingPipeline(
    storage_path=STORAGE_PATH
)


result = indexing_pipeline.run(
    file_paths=FILE_PATHS
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n" + "=" * 80)
print("MULTI-DOCUMENT INDEXING")
print("=" * 80)

print(
    f"Files Processed: "
    f"{result['files']}"
)

print(
    f"Documents Loaded: "
    f"{result['documents']}"
)

print(
    f"Chunks Created: "
    f"{result['chunks']}"
)

print(
    f"Vectors Indexed: "
    f"{result['vectors']}"
)