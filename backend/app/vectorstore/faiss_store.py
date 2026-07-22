"""
Thin wrapper around a FAISS flat index plus a parallel metadata list.

We use IndexFlatIP (inner product) over normalized vectors, which is
equivalent to cosine similarity and is exact (no approximation) -- the
right choice until the corpus is large enough that exact search is too
slow, at which point swapping in IndexIVFFlat or HNSW is a drop-in
change confined to this file.

Metadata (chunk text, source filename, page, doc_id) lives in a plain
list kept in lockstep with the FAISS index by position, and is
persisted alongside it with pickle.
"""
import pickle
import threading
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np

from app.config import settings


class FaissStore:
    def __init__(self, dim: int):
        self._lock = threading.Lock()
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = []
        self._load_if_exists()

    def _load_if_exists(self):
        if settings.index_path.exists() and settings.metadata_path.exists():
            self.index = faiss.read_index(str(settings.index_path))
            with open(settings.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

    def _persist(self):
        faiss.write_index(self.index, str(settings.index_path))
        with open(settings.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        with self._lock:
            self.index.add(vectors)
            self.metadata.extend(metadatas)
            self._persist()

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        if self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(query_vector.reshape(1, -1), top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        docs: Dict[str, Dict[str, Any]] = {}
        for m in self.metadata:
            doc_id = m["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = {"doc_id": doc_id, "filename": m["source"], "num_chunks": 0}
            docs[doc_id]["num_chunks"] += 1
        return list(docs.values())


_store_singleton: FaissStore = None


def get_store(dim: int) -> FaissStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = FaissStore(dim)
    return _store_singleton
