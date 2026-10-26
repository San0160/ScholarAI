from types import SimpleNamespace

from ai_research_assistant.pipeline.answer_service import answer_question


class FakeRetrieval:
    def __init__(self, results=None, chunks=None):
        self.results = results or []
        self.chunks = chunks or []
        self.run_calls = 0

    def run(self, question, metadata_filters=None, top_k=None):
        self.run_calls += 1
        return self.results

    def get_document_chunks(self, metadata_filters):
        return self.chunks


class FakeGeneration:
    def __init__(self, sources, cited):
        self.sources = sources
        self.cited = cited
        self.received = None

    def select_overview_documents(self, chunks):
        return list(chunks)

    def run(self, query, documents):
        self.received = documents
        return {
            "answer": "an answer",
            "context": "the context",
            "sources": self.sources,
            "raw_citation_validation": {"cited_sources": self.cited},
        }


SOURCES = {1: {"filename": "a.pdf", "page": 1}, 2: {"filename": "a.pdf", "page": 4}}


def test_returns_none_when_nothing_retrieved():
    assert answer_question(FakeRetrieval(), FakeGeneration(SOURCES, []), "What is X?") is None


def test_shows_only_the_sources_the_model_cited():
    retrieval = FakeRetrieval(results=[SimpleNamespace(document="doc")])
    outcome = answer_question(retrieval, FakeGeneration(SOURCES, [2]), "What is X?")
    assert outcome.model_cited is True
    assert outcome.citations == [SOURCES[2]]
    assert outcome.context_sources == [SOURCES[1], SOURCES[2]]


def test_falls_back_to_all_sources_when_model_cited_nothing():
    retrieval = FakeRetrieval(results=[SimpleNamespace(document="doc")])
    outcome = answer_question(retrieval, FakeGeneration(SOURCES, []), "What is X?")
    assert outcome.model_cited is False
    assert outcome.citations == [SOURCES[1], SOURCES[2]]


def test_ignores_cited_numbers_that_are_not_real_sources():
    retrieval = FakeRetrieval(results=[SimpleNamespace(document="doc")])
    outcome = answer_question(retrieval, FakeGeneration(SOURCES, [9]), "What is X?")
    assert outcome.model_cited is False


def test_summary_question_reads_the_whole_document_without_searching():
    retrieval = FakeRetrieval(chunks=["chunk1", "chunk2"])
    generation = FakeGeneration(SOURCES, [1])
    outcome = answer_question(retrieval, generation, "Summarize this paper", metadata_filters={"filename": "a.pdf"})
    assert generation.received == ["chunk1", "chunk2"]
    assert retrieval.run_calls == 0
    assert outcome is not None


def test_summary_question_without_chunks_falls_back_to_search():
    retrieval = FakeRetrieval(results=[SimpleNamespace(document="doc")], chunks=[])
    generation = FakeGeneration(SOURCES, [1])
    answer_question(retrieval, generation, "Summarize this paper", metadata_filters={"filename": "a.pdf"})
    assert retrieval.run_calls == 1
    assert generation.received == ["doc"]
