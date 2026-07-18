from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _sentence_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _nli_pipeline():
    from transformers import pipeline

    return pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")


def embed_fn(text):
    vector = _sentence_model().encode(text, convert_to_numpy=True)
    return vector.astype("float32")


def embedding_dim():
    return _sentence_model().get_sentence_embedding_dimension()


def nli_fn(premise, hypothesis):
    result = _nli_pipeline()(f"{premise}", text_pair=hypothesis)
    return result[0]["label"].lower()
