from ai_research_assistant.pipeline.indexing_pipeline import IndexingPipeline
from ai_research_assistant.pipeline.query_pipeline import QueryPipeline


# -------------------------
# Index document
# -------------------------

indexing_pipeline = IndexingPipeline()

result = indexing_pipeline.run(
    "data/documents/GAN_case_study.pdf"
)

print("Indexing complete:", result)
print("Vectors in FAISS:", indexing_pipeline.vector_store.index.ntotal)


# -------------------------
# Ask question
# -------------------------

query_pipeline = QueryPipeline()

query = "What is the main contribution of this paper?"

results = query_pipeline.retrieval_pipeline.run(query)

print("\nRetrieved results:", len(results))

for result in results:
    print("\nScore:", result.score)
    print("Content:", result.document.page_content[:500])

answer = query_pipeline.run(query)

print("\nAnswer:")
print(answer)