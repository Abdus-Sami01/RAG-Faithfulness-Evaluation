import re


def extract_claims(text):
    fragments = re.split(r"(?<=[.!?])\s+", text.strip())
    return [f.strip() for f in fragments if f.strip() and f.strip() != "."]


def check_faithfulness(claims, contexts, nli_fn):
    results = []
    for claim in claims:
        supported = any(nli_fn(context, claim) == "entailment" for context in contexts)
        results.append(supported)
    return results
