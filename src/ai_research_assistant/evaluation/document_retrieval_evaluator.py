class DocumentRetrievalEvaluator:
    """Document-level retrieval evaluation: measures whether retrieval
    correctly favors the expected SOURCE FILE for a question, rather than
    the exact chunk (see RetrievalEvaluator for the chunk-level version).
    Intended for a multi-document corpus where each question has one
    correct source document among several candidates, and what matters is
    whether retrieval confuses documents, not chunk boundaries.
    """

    def __init__(self, pipeline, k: int = 3):
        self.pipeline = pipeline
        self.k = k

    def evaluate(self, questions: list[dict]) -> dict:
        hits = []
        reciprocal_ranks = []
        precisions = []
        detailed_results = []

        for question in questions:
            query = question["question"]
            expected_source = question["expected_source"]

            results = self.pipeline.run(query)
            results = results[: self.k]

            retrieved_sources = [
                result.document.metadata.get("filename") for result in results
            ]

            hit = expected_source in retrieved_sources
            hits.append(hit)

            reciprocal_rank = 0.0
            for rank, filename in enumerate(retrieved_sources, start=1):
                if filename == expected_source:
                    reciprocal_rank = 1.0 / rank
                    break
            reciprocal_ranks.append(reciprocal_rank)

            correct_count = sum(1 for f in retrieved_sources if f == expected_source)
            precision = correct_count / len(retrieved_sources) if retrieved_sources else 0.0
            precisions.append(precision)

            detailed_results.append({
                "question": query,
                "expected_source": expected_source,
                "retrieved_sources": retrieved_sources,
                "hit": hit,
            })

        return {
            "hit_rate": sum(hits) / len(hits) if hits else 0.0,
            "precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
            "details": detailed_results,
        }