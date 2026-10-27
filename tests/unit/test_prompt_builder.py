from ai_research_assistant.generation.prompt_builder import PromptBuilder


def test_build_returns_system_then_user_message():
    messages = PromptBuilder().build(query="Which optimizer was used?", context="[Source 1]\nAdam was used.")

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == PromptBuilder.SYSTEM_PROMPT


def test_user_message_contains_the_context_and_the_question():
    messages = PromptBuilder().build(query="Which optimizer was used?", context="[Source 1]\nAdam was used.")
    user_message = messages[1]["content"]

    assert "Adam was used." in user_message
    assert "Which optimizer was used?" in user_message
    assert user_message.index("Adam was used.") < user_message.index("Which optimizer was used?")


def test_system_prompt_tells_the_model_how_to_cite():
    assert "[Source N]" in PromptBuilder.SYSTEM_PROMPT
