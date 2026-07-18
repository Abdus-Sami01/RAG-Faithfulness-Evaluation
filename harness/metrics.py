def precision_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def mrr(retrieved_lists, relevant_sets):
    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        rank = next((i + 1 for i, doc in enumerate(retrieved) if doc in relevant), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def faithfulness_rate(claim_supported_flags):
    if not claim_supported_flags:
        return 0.0
    return sum(claim_supported_flags) / len(claim_supported_flags)
