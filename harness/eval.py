import json

from harness.metrics import precision_at_k, recall_at_k, mrr, faithfulness_rate
from harness.faithfulness import check_faithfulness, extract_claims


def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(index, eval_queries, k):
    retrieved_lists = []
    relevant_sets = []
    per_query = []
    for item in eval_queries:
        retrieved = index.search(item["query"], k=k)
        relevant = set(item["relevant_doc_ids"])
        retrieved_lists.append(retrieved)
        relevant_sets.append(relevant)
        per_query.append({
            "query": item["query"],
            "retrieved": retrieved,
            "precision_at_k": precision_at_k(retrieved, relevant, k),
            "recall_at_k": recall_at_k(retrieved, relevant, k),
        })
    return {
        "per_query": per_query,
        "mean_precision_at_k": sum(q["precision_at_k"] for q in per_query) / len(per_query),
        "mean_recall_at_k": sum(q["recall_at_k"] for q in per_query) / len(per_query),
        "mrr": mrr(retrieved_lists, relevant_sets),
    }


def evaluate_faithfulness(probes, docs_by_id, nli_fn):
    results = []
    for probe in probes:
        contexts = [
            sentence
            for doc_id in probe["context_doc_ids"]
            for sentence in extract_claims(docs_by_id[doc_id])
        ]
        supported = check_faithfulness([probe["claim"]], contexts, nli_fn)[0]
        results.append({
            "claim": probe["claim"],
            "expected_supported": probe["expected_supported"],
            "predicted_supported": supported,
            "correct": supported == probe["expected_supported"],
        })
    return {
        "per_probe": results,
        "faithfulness_rate": faithfulness_rate([r["predicted_supported"] for r in results]),
        "detector_accuracy": sum(r["correct"] for r in results) / len(results),
    }
