from harness.metrics import precision_at_k, recall_at_k, mrr, faithfulness_rate


def test_precision_at_k_all_relevant():
    assert precision_at_k(retrieved=["a", "b"], relevant={"a", "b"}, k=2) == 1.0


def test_precision_at_k_half_relevant():
    assert precision_at_k(retrieved=["a", "x"], relevant={"a", "b"}, k=2) == 0.5


def test_precision_at_k_truncates_to_k():
    assert precision_at_k(retrieved=["a", "b", "x"], relevant={"a", "b"}, k=2) == 1.0


def test_recall_at_k_finds_all_relevant():
    assert recall_at_k(retrieved=["a", "b"], relevant={"a", "b"}, k=2) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(retrieved=["a", "x"], relevant={"a", "b"}, k=2) == 0.5


def test_recall_at_k_empty_relevant_is_zero():
    assert recall_at_k(retrieved=["a"], relevant=set(), k=1) == 0.0


def test_mrr_first_hit_at_rank_one():
    assert mrr([["a", "b"]], [{"a"}]) == 1.0


def test_mrr_first_hit_at_rank_two():
    assert mrr([["x", "a"]], [{"a"}]) == 0.5


def test_mrr_no_hit_is_zero():
    assert mrr([["x", "y"]], [{"a"}]) == 0.0


def test_mrr_averages_across_queries():
    assert mrr([["a", "x"], ["x", "b"]], [{"a"}, {"b"}]) == 0.75


def test_faithfulness_rate_all_supported():
    assert faithfulness_rate([True, True, True]) == 1.0


def test_faithfulness_rate_none_supported():
    assert faithfulness_rate([False, False]) == 0.0


def test_faithfulness_rate_mixed():
    assert faithfulness_rate([True, False, True, False]) == 0.5


def test_faithfulness_rate_empty_claims_is_zero():
    assert faithfulness_rate([]) == 0.0
