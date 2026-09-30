from ai_research_assistant.utils.tokenizer_utils import get_token_counter

count_tokens = get_token_counter("openai/gpt-oss-120b")
sample = "The Transformer is the first sequence transduction model based entirely on attention."
print(count_tokens(sample) / len(sample.split()))