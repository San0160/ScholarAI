class RetrievalEvaluator:

    def __init__(
        self,
        pipeline,
        k: int = 3
    ):

        self.pipeline = pipeline
        self.k = k

    def evaluate(
        self,
        questions
    ):

        recall_scores = []
        precision_scores = []
        reciprocal_ranks = []

        detailed_results = []

        for question in questions:

            query = question["question"]
            expected_chunks = question["expected_chunks"]

            results = self.pipeline.run(
                query
            )

            results = results[:self.k]

            retrieved_chunks = [
                result.document.metadata.get(
                    "chunk_id"
                )
                for result in results
            ]

            expected = set(
                expected_chunks
            )

            retrieved = set(
                retrieved_chunks
            )

            # Recall@K
            if expected:

                recall = len(
                    expected.intersection(retrieved)
                ) / len(expected)

            else:

                recall = 0.0

            # Precision@K
            if retrieved:

                precision = len(
                    expected.intersection(retrieved)
                ) / len(retrieved)

            else:

                precision = 0.0

            # Reciprocal Rank
            reciprocal_rank = 0.0

            for rank, chunk_id in enumerate(
                retrieved_chunks,
                start=1
            ):

                if chunk_id in expected:

                    reciprocal_rank = 1.0 / rank
                    break

            recall_scores.append(
                recall
            )

            precision_scores.append(
                precision
            )

            reciprocal_ranks.append(
                reciprocal_rank
            )

            detailed_results.append(
                {
                    "question": query,
                    "expected_chunks": expected_chunks,
                    "retrieved_chunks": retrieved_chunks,
                    "hit": recall > 0
                }
            )

        return {
            "recall": (
                sum(recall_scores)
                / len(recall_scores)
            ),

            "precision": (
                sum(precision_scores)
                / len(precision_scores)
            ),

            "mrr": (
                sum(reciprocal_ranks)
                / len(reciprocal_ranks)
            ),

            "details": detailed_results
        }