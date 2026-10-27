from ai_research_assistant.entity.document import Document
from ai_research_assistant.generation.context_builder import ContextBuilder


def word_count(text: str) -> int:
    """A stand-in token counter, so the tests need no tokenizer download."""
    return len(text.split())


def make_builder(max_context_tokens: int) -> ContextBuilder:
    return ContextBuilder(token_counter=word_count, max_context_tokens=max_context_tokens)


def make_document(text: str, page: int, chunk_id: str = "a.pdf_0") -> Document:
    return Document(page_content=text, metadata={"filename": "a.pdf", "page": page, "chunk_id": chunk_id})


# Each formatted part adds 6 words of labels ("[Source", "N]", "Document:", "a.pdf", "Page:", "N"),
# so a 4-word chunk costs 10 "tokens".
FOUR_WORDS = "one two three four"


def test_build_labels_each_source_and_maps_it_to_its_page():
    documents = [make_document(FOUR_WORDS, page=7, chunk_id="a.pdf_0"), make_document(FOUR_WORDS, page=9, chunk_id="a.pdf_1")]

    context, source_map = make_builder(100).build(documents)

    assert "[Source 1]" in context and "[Source 2]" in context
    assert "Page: 9" in context
    assert source_map[1]["page"] == 7
    assert source_map[2]["page"] == 9
    assert source_map[2]["chunk_id"] == "a.pdf_1"
    assert source_map[1]["filename"] == "a.pdf"


def test_build_stops_when_the_budget_is_reached():
    documents = [make_document(FOUR_WORDS, page=page) for page in (1, 2, 3)]

    context, source_map = make_builder(25).build(documents)  # room for two 10-token parts

    assert set(source_map) == {1, 2}
    assert "[Source 3]" not in context


def test_build_with_no_documents_returns_empty_context():
    assert make_builder(100).build([]) == ("", {})


def test_overview_keeps_everything_when_it_fits():
    documents = [make_document(FOUR_WORDS, page=page) for page in range(1, 4)]

    assert make_builder(100).select_for_overview(documents) == documents


def test_overview_spreads_chunks_evenly_when_too_big():
    documents = [make_document(FOUR_WORDS, page=page) for page in range(1, 11)]  # 10 x 10 = 100 tokens

    chosen = make_builder(45).select_for_overview(documents)

    assert 1 < len(chosen) < len(documents)
    assert chosen[0] is documents[0] and chosen[-1] is documents[-1]  # covers start and end of the document
    assert [d.metadata["page"] for d in chosen] == sorted(d.metadata["page"] for d in chosen)  # keeps order
    assert sum(10 for _ in chosen) <= 45


def test_overview_with_no_documents_or_nothing_that_fits():
    assert make_builder(100).select_for_overview([]) == []
    assert make_builder(5).select_for_overview([make_document(FOUR_WORDS, page=1)]) == []
