from harness.eval import evaluate_retrieval, evaluate_faithfulness


class FakeIndex:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    def search(self, query_text, k):
        return self.results_by_query[query_text][:k]


def test_evaluate_retrieval_reports_per_query_and_aggregate_metrics():
    index = FakeIndex({
        "q1": ["a", "x"],
        "q2": ["b", "y"],
    })
    eval_queries = [
        {"query": "q1", "relevant_doc_ids": ["a"]},
        {"query": "q2", "relevant_doc_ids": ["z"]},
    ]

    result = evaluate_retrieval(index, eval_queries, k=2)

    assert result["per_query"][0]["precision_at_k"] == 0.5
    assert result["per_query"][1]["precision_at_k"] == 0.0
    assert result["mean_precision_at_k"] == 0.25
    assert result["mrr"] == 0.5


def test_evaluate_faithfulness_flags_correct_and_incorrect_predictions():
    docs_by_id = {"doc1": "Leave accrues at 1.5 days per month."}
    probes = [
        {"context_doc_ids": ["doc1"], "claim": "true claim", "expected_supported": True},
        {"context_doc_ids": ["doc1"], "claim": "false claim", "expected_supported": False},
    ]

    def fake_nli(premise, hypothesis):
        return "entailment" if hypothesis == "true claim" else "contradiction"

    result = evaluate_faithfulness(probes, docs_by_id, fake_nli)

    assert result["per_probe"][0]["predicted_supported"] is True
    assert result["per_probe"][0]["correct"] is True
    assert result["per_probe"][1]["predicted_supported"] is False
    assert result["per_probe"][1]["correct"] is True
    assert result["detector_accuracy"] == 1.0


def test_evaluate_faithfulness_checks_claim_against_individual_sentences_not_whole_doc():
    docs_by_id = {"doc1": "Bananas are yellow. Leave accrues at 1.5 days per month."}
    probes = [{"context_doc_ids": ["doc1"], "claim": "Leave accrues at 1.5 days per month.", "expected_supported": True}]

    def exact_sentence_match_only(premise, hypothesis):
        return "entailment" if premise.strip() == hypothesis.strip() else "neutral"

    result = evaluate_faithfulness(probes, docs_by_id, exact_sentence_match_only)

    assert result["per_probe"][0]["predicted_supported"] is True


def test_evaluate_faithfulness_marks_wrong_predictions_incorrect():
    docs_by_id = {"doc1": "context"}
    probes = [{"context_doc_ids": ["doc1"], "claim": "claim", "expected_supported": True}]

    def always_unsupported(premise, hypothesis):
        return "neutral"

    result = evaluate_faithfulness(probes, docs_by_id, always_unsupported)

    assert result["per_probe"][0]["correct"] is False
    assert result["detector_accuracy"] == 0.0
