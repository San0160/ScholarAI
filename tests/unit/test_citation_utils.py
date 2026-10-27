from ai_research_assistant.utils.citation_utils import SOURCE_BLOCK_PATTERN, extract_source_ids


def cited_ids(text: str) -> list[int]:
    ids = []
    for block in SOURCE_BLOCK_PATTERN.findall(text):
        ids.extend(extract_source_ids(block))
    return ids


def test_square_bracket_citation():
    assert cited_ids("Adam was used. [Source 2]") == [2]


def test_several_ids_in_one_bracket():
    assert cited_ids("Both agree. [Sources 1, 3 and 4]") == [1, 3, 4]


def test_page_number_after_a_colon_is_not_a_source_id():
    assert cited_ids("Stated here. [Source 2: attention.pdf, p. 7]") == [2]


def test_full_width_brackets_are_recognised():
    # Some models write 【Source 1】 instead of [Source 1].
    assert cited_ids("It showed mode collapse【Source 1】.") == [1]


def test_full_width_markers_are_removed_from_the_visible_answer():
    cleaned = SOURCE_BLOCK_PATTERN.sub("", "It collapsed【Source 1】 and recovered [Source 2].")
    assert "Source" not in cleaned and "【" not in cleaned


def test_the_documents_own_reference_markers_are_not_citations():
    assert cited_ids("The model used Adam [20].") == []
