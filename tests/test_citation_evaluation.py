import json
import numpy as np

from ai_research_assistant.config.configuration import ConfigurationManager
from ai_research_assistant.embeddings.embedding_factory import EmbeddingFactory
from ai_research_assistant.pipeline.retrieval_pipeline import RetrievalPipeline
from ai_research_assistant.pipeline.generation_pipeline import GenerationPipeline
from ai_research_assistant.citation.citation_validator import CitationValidator  


config = ConfigurationManager().config
QUESTION_FILE = "src/ai_research_assistant/evaluation/question_attention.json"

with open(QUESTION_FILE, "r", encoding="utf-8") as file:
    questions = json.load(file)

embedder = EmbeddingFactory.create_embedding()

retrieval_pipeline = RetrievalPipeline()
generation_pipeline = GenerationPipeline(
    model_name=config.llm.model,
    embedder=embedder
)

# Standalone validator, independent of the pipeline's internal citation_validation
raw_citation_validator = CitationValidator()

results = []

for index, item in enumerate(questions, start=1):

    question = item["question"]
    expected_chunks = set(item.get("expected_chunks", []))

    retrieved_results = retrieval_pipeline.run(question)
    documents = [result.document for result in retrieved_results]

    result = generation_pipeline.run(query=question, documents=documents)

    raw_answer = result["answer"]
    source_map = result["sources"]

    # ------------------------------------------------------------
    # RAW LLM SELF-CITATION COMPLIANCE
    # Parses [Source N] directly from the LLM's own generated text.
    # Bypasses CitationMatcher entirely — measures whether the model
    # itself followed the citation instruction, not whether a
    # fallback system can infer the right source after the fact.
    # ------------------------------------------------------------
    raw_validation = raw_citation_validator.validate(
        answer=raw_answer,
        source_map=source_map
    )

    print(f"\nQuestion {index}: {question}")
    print(f"RAW ANSWER TEXT:\n{raw_answer}")
    print(f"Citations found by regex: {raw_validation['cited_sources']}")

    raw_cited_chunk_ids = {
        source_map[n]["chunk_id"]
        for n in raw_validation["cited_sources"]
        if n in source_map
    }

    raw_has_citation = raw_validation["has_citations"]
    raw_all_valid = raw_validation["all_citations_valid"]
    raw_hit = len(raw_cited_chunk_ids.intersection(expected_chunks)) > 0

    if raw_cited_chunk_ids:
        raw_precision = len(raw_cited_chunk_ids.intersection(expected_chunks)) / len(raw_cited_chunk_ids)
    else:
        raw_precision = 0.0

    results.append({
        "question": question,
        "raw_has_citation": raw_has_citation,
        "raw_all_valid": raw_all_valid,
        "raw_hit": raw_hit,
        "raw_precision": raw_precision,
        "raw_cited_sources": raw_validation["cited_sources"],
        "raw_invalid_citations": raw_validation["invalid_citations"],
    })

    print(f"\nQuestion {index}: {question}")
    print(f"Raw answer citations found: {raw_validation['cited_sources']}")
    print(f"Raw has_citation: {raw_has_citation} | valid: {raw_all_valid} | hit: {raw_hit}")


# --------------------------------------------------
# Overall Raw Compliance Metrics
# --------------------------------------------------

total = len(results)

raw_coverage = sum(r["raw_has_citation"] for r in results) / total
raw_validity = sum(r["raw_all_valid"] for r in results) / total
raw_hit_rate = sum(r["raw_hit"] for r in results) / total
raw_avg_precision = sum(r["raw_precision"] for r in results) / total

print("\n" + "=" * 80)
print("LLM RAW SELF-CITATION COMPLIANCE (bypasses CitationMatcher)")
print("=" * 80)
print(f"Coverage (has >=1 [Source N]):     {raw_coverage:.4f}")
print(f"Validity (all cited N are real):   {raw_validity:.4f}")
print(f"Hit Rate (cited chunk == correct): {raw_hit_rate:.4f}")
print(f"Average Precision:                 {raw_avg_precision:.4f}")