from dotenv import load_dotenv
load_dotenv()

from ai_research_assistant.pipeline.indexing_pipeline import IndexingPipeline

FILE_PATHS = [
    "data/documents/attention.pdf",
    "data/documents/deeplab.pdf",
    "data/documents/Image_Segmentation.pdf",
    "data/documents/Text_Summariser.pdf",
]

STORAGE_PATH = "artifacts/vector_store/multi_document"

indexing_pipeline = IndexingPipeline(storage_path=STORAGE_PATH)
result = indexing_pipeline.run(file_paths=FILE_PATHS, rebuild=True)
print(result)