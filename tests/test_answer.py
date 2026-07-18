from harness.answer import extractive_answer


def test_picks_sentence_with_most_query_word_overlap():
    contexts = ["The capital of France is Paris. It is known for the Eiffel Tower."]
    answer = extractive_answer("What is the capital of France?", contexts, top_n=1)
    assert answer == "The capital of France is Paris."


def test_ranks_across_multiple_contexts():
    contexts = [
        "Bananas are yellow fruit.",
        "The capital of France is Paris.",
    ]
    answer = extractive_answer("What is the capital of France?", contexts, top_n=1)
    assert answer == "The capital of France is Paris."


def test_top_n_returns_multiple_sentences_joined():
    contexts = ["The capital of France is Paris. Paris hosts the Louvre museum."]
    answer = extractive_answer("Tell me about Paris", contexts, top_n=2)
    assert answer == "The capital of France is Paris. Paris hosts the Louvre museum."


def test_no_context_returns_empty_string():
    assert extractive_answer("anything", [], top_n=1) == ""
