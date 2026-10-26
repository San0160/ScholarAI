from ai_research_assistant.evaluation.answer_checks import (
    average,
    citation_check,
    fact_check,
    normalize,
    strip_markers,
)


def test_strip_markers_removes_source_and_reference_markers():
    text = "The model has 6 layers. [Source 6] It uses Adam [20] and more [3, 4]."
    cleaned = strip_markers(text)
    assert "[Source 6]" not in cleaned
    assert "[20]" not in cleaned
    assert "[3, 4]" not in cleaned
    assert "6 layers" in cleaned


def test_normalize_folds_ligatures_dashes_and_case():
    assert normalize("Multi–Head  \n Attention ﬁnal") == "multi-head attention final"


def test_fact_check_all_found():
    result = fact_check("There are six encoder layers and six decoder layers.", [["six", "6"], "encoder", "decoder"])
    assert result["fact_recall"] == 1.0
    assert result["missing_facts"] == []


def test_fact_check_reports_missing():
    result = fact_check("It uses an encoder.", ["encoder", "decoder"])
    assert result["fact_recall"] == 0.5
    assert result["missing_facts"] == ["decoder"]


def test_fact_check_whole_word_does_not_match_inside_numbers():
    # "6" must not be found in "16" or "0.6"
    assert fact_check("There are 16 layers and a rate of 0.6.", [["6"]])["fact_recall"] == 0.0
    assert fact_check("There are 6 layers.", [["6"]])["fact_recall"] == 1.0


def test_fact_check_ignores_numbers_inside_citation_markers():
    assert fact_check("The layers are stacked. [Source 6]", [["6"]])["fact_recall"] == 0.0


def test_fact_check_prefix_wildcard():
    assert fact_check("It allows parallelization.", ["parallel*"])["fact_recall"] == 1.0
    assert fact_check("It allows parallelization.", ["parallel"])["fact_recall"] == 0.0


def test_fact_check_decimal_number():
    assert fact_check("It reached a BLEU score of 28.4 on the test set.", ["28.4"])["fact_recall"] == 1.0
    assert fact_check("It reached a BLEU score of 28.", ["28.4"])["fact_recall"] == 0.0


def test_fact_check_empty_list_has_no_score():
    assert fact_check("anything", [])["fact_recall"] is None


def test_citation_check_right_page():
    result = citation_check(cited_pages=[3, 4], gold_pages=[3], context_pages=[3, 4, 5])
    assert result["citation_hit"] is True
    assert result["citation_precision"] == 0.5
    assert result["gold_in_context"] is True


def test_citation_check_wrong_page_but_gold_was_in_prompt():
    result = citation_check(cited_pages=[5], gold_pages=[3], context_pages=[3, 5])
    assert result["citation_hit"] is False
    assert result["citation_precision"] == 0.0
    assert result["gold_in_context"] is True


def test_citation_check_nothing_cited():
    result = citation_check(cited_pages=[], gold_pages=[3], context_pages=[1, 2])
    assert result["citation_hit"] is False
    assert result["citation_precision"] == 0.0
    assert result["gold_in_context"] is False


def test_average_skips_missing_and_counts_booleans():
    rows = [{"x": True}, {"x": False}, {"x": None}, {}]
    assert average(rows, "x") == 0.5
    assert average([{}], "x") is None
