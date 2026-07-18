import numpy as np

from harness.index import VectorIndex

VECTORS = {
    "doc-close": np.array([1.0, 0.0], dtype="float32"),
    "doc-far": np.array([0.0, 1.0], dtype="float32"),
    "query-near-close": np.array([0.9, 0.1], dtype="float32"),
}


def fake_embed(text):
    return VECTORS[text]


def test_search_returns_closest_doc_first():
    index = VectorIndex(embed_fn=fake_embed, dim=2)
    index.build(doc_ids=["doc-close", "doc-far"], doc_texts=["doc-close", "doc-far"])

    results = index.search("query-near-close", k=2)

    assert results[0] == "doc-close"
    assert results[1] == "doc-far"


def test_search_respects_k():
    index = VectorIndex(embed_fn=fake_embed, dim=2)
    index.build(doc_ids=["doc-close", "doc-far"], doc_texts=["doc-close", "doc-far"])

    results = index.search("query-near-close", k=1)

    assert results == ["doc-close"]


def test_build_rejects_mismatched_lengths():
    index = VectorIndex(embed_fn=fake_embed, dim=2)
    try:
        index.build(doc_ids=["a", "b"], doc_texts=["only-one"])
        assert False, "expected ValueError"
    except ValueError:
        pass
