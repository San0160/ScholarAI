from ai_research_assistant.pipeline.indexing_pipeline import IndexingPipeline
from ai_research_assistant.pipeline.query_pipeline import QueryPipeline


# -------------------------
# Index document
# -------------------------

indexing_pipeline = IndexingPipeline()

result = indexing_pipeline.run(
    "data/documents/attention.pdf"
)

print("Indexing complete:", result)

print(
    "Vectors in FAISS:",
    indexing_pipeline.vector_store.index.ntotal
)


# -------------------------
# Ask question
# -------------------------
'''
query_pipeline = QueryPipeline()

answer = query_pipeline.run(
    "What is the main contribution of this paper?"
)

print("\nAnswer:")
print(answer)
'''