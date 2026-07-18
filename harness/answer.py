import re

from harness.faithfulness import extract_claims

_STOPWORDS = {"the", "is", "of", "a", "an", "to", "for", "and", "it", "in", "on", "what", "tell", "me", "about"}


def extractive_answer(query, contexts, top_n):
    query_words = _content_words(query)
    sentences = [s for ctx in contexts for s in extract_claims(ctx)]
    if not sentences:
        return ""
    scored = [(len(query_words & _content_words(s)), i, s) for i, s in enumerate(sentences)]
    scored.sort(key=lambda item: (-item[0], item[1]))
    top = sorted(scored[:top_n], key=lambda item: item[1])
    return " ".join(s for _, _, s in top)


def _content_words(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return {w for w in words if w not in _STOPWORDS}
