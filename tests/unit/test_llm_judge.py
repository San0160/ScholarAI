from ai_research_assistant.evaluation.llm_judge import LLMJudge, parse_judge_reply


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


GOOD = '{"faithfulness": 5, "faithfulness_reason": "ok", "correctness": 4, "correctness_reason": "minor gap"}'


def test_parse_plain_json():
    parsed = parse_judge_reply(GOOD, expect_correctness=True)
    assert parsed["faithfulness"] == 5
    assert parsed["correctness"] == 4


def test_parse_json_wrapped_in_text_and_code_fence():
    reply = "Here is my grade:\n```json\n" + GOOD + "\n```"
    assert parse_judge_reply(reply, expect_correctness=True)["correctness"] == 4


def test_parse_without_reference_ignores_correctness():
    parsed = parse_judge_reply(GOOD, expect_correctness=False)
    assert parsed["faithfulness"] == 5
    assert parsed["correctness"] is None


def test_parse_rejects_out_of_range_and_garbage():
    assert parse_judge_reply('{"faithfulness": 9, "correctness": 3}', expect_correctness=True) is None
    assert parse_judge_reply("no json here", expect_correctness=True) is None
    assert parse_judge_reply('{"faithfulness": 5}', expect_correctness=True) is None  # reference given, correctness missing


def test_grade_passes_reference_and_strips_markers():
    llm = FakeLLM([GOOD])
    result = LLMJudge(llm).grade("Q?", "The answer is 6. [Source 6]", "ctx text", reference="ref text")
    user_message = llm.calls[0][1]["content"]
    assert "ref text" in user_message
    assert "ctx text" in user_message
    assert "[Source 6]" not in user_message
    assert result["faithfulness"] == 5 and result["judge_error"] is None


def test_grade_retries_once_on_bad_json():
    llm = FakeLLM(["sorry, I cannot", GOOD])
    result = LLMJudge(llm).grade("Q?", "A", "ctx", reference="ref")
    assert len(llm.calls) == 2
    assert result["correctness"] == 4


def test_grade_gives_up_after_two_bad_replies():
    llm = FakeLLM(["nope", "still nope"])
    result = LLMJudge(llm).grade("Q?", "A", "ctx", reference="ref")
    assert result["faithfulness"] is None
    assert result["judge_error"]
