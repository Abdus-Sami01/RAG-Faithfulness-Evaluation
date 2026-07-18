import numpy as np
import faiss


class VectorIndex:
    def __init__(self, embed_fn, dim):
        self.embed_fn = embed_fn
        self.dim = dim
        self.doc_ids = []
        self.faiss_index = faiss.IndexFlatIP(dim)

    def build(self, doc_ids, doc_texts):
        if len(doc_ids) != len(doc_texts):
            raise ValueError("doc_ids and doc_texts must be the same length")
        vectors = np.stack([self._normalize(self.embed_fn(text)) for text in doc_texts])
        self.faiss_index.add(vectors)
        self.doc_ids = list(doc_ids)

    def search(self, query_text, k):
        query_vector = self._normalize(self.embed_fn(query_text)).reshape(1, -1)
        _, indices = self.faiss_index.search(query_vector, k)
        return [self.doc_ids[i] for i in indices[0] if i != -1]

    @staticmethod
    def _normalize(vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
