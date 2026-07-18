from harness.faithfulness import extract_claims, check_faithfulness


def test_extract_claims_splits_on_sentence_boundaries():
    assert extract_claims("The sky is blue. Water is wet.") == ["The sky is blue.", "Water is wet."]


def test_extract_claims_strips_whitespace():
    assert extract_claims("  One claim.  ") == ["One claim."]


def test_extract_claims_ignores_empty_fragments():
    assert extract_claims("First. . Second.") == ["First.", "Second."]


def test_check_faithfulness_supported_when_any_context_entails():
    def fake_nli(premise, hypothesis):
        return "entailment" if premise == "ctx-match" else "neutral"

    result = check_faithfulness(
        claims=["claim a"],
        contexts=["ctx-other", "ctx-match"],
        nli_fn=fake_nli,
    )
    assert result == [True]


def test_check_faithfulness_unsupported_when_no_context_entails():
    def fake_nli(premise, hypothesis):
        return "neutral"

    result = check_faithfulness(claims=["claim a"], contexts=["ctx1"], nli_fn=fake_nli)
    assert result == [False]


def test_check_faithfulness_contradiction_counts_as_unsupported():
    def fake_nli(premise, hypothesis):
        return "contradiction"

    result = check_faithfulness(claims=["claim a"], contexts=["ctx1"], nli_fn=fake_nli)
    assert result == [False]


def test_check_faithfulness_evaluates_each_claim_independently():
    def fake_nli(premise, hypothesis):
        return "entailment" if hypothesis == "true claim" else "neutral"

    result = check_faithfulness(
        claims=["true claim", "false claim"],
        contexts=["ctx1"],
        nli_fn=fake_nli,
    )
    assert result == [True, False]
