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
            # Keep self.dim in sync with whatever was persisted on disk
            # so callers that check store.dim get the real vector size.
            self.dim = self.index.d

    def _persist(self):
        faiss.write_index(self.index, str(settings.index_path))
        with open(settings.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        # Issue 2: guard against a dim mismatch caused by swapping embedding
        # models after the index was already created with a different dim.
        if vectors.shape[1] != self.index.d:
            raise ValueError(
                f"Vector dimension mismatch: index expects {self.index.d}, "
                f"got {vectors.shape[1]}. Delete the persisted index "
                f"({settings.index_path}) to rebuild with the new model."
            )
        with self._lock:
            self.index.add(vectors)
            self.metadata.extend(metadatas)
            self._persist()

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        doc_id: str = None,   # Issue 4: None = search whole corpus
    ) -> List[Tuple[Dict[str, Any], float]]:
        if self.index.ntotal == 0:
            return []
        # Issue 2: same guard on query side
        if query_vector.shape[0] != self.index.d:
            raise ValueError(
                f"Query vector dimension mismatch: index expects {self.index.d}, "
                f"got {query_vector.shape[0]}."
            )

        if doc_id is None:
            # Fast path: search the full index.
            scores, ids = self.index.search(query_vector.reshape(1, -1), top_k)
            results = []
            for score, idx in zip(scores[0], ids[0]):
                if idx == -1:
                    continue
                results.append((self.metadata[idx], float(score)))
            return results

        # Filtered path: only consider chunks belonging to doc_id.
        # We over-fetch and post-filter because FAISS flat indexes don't
        # support predicate pushdown. Fine at demo scale.
        candidate_indices = [
            i for i, m in enumerate(self.metadata) if m["doc_id"] == doc_id
        ]
        if not candidate_indices:
            return []

        candidate_vectors = np.array(
            [self.index.reconstruct(i) for i in candidate_indices], dtype="float32"
        )
        # Build a temporary index over just this document's vectors.
        tmp_index = faiss.IndexFlatIP(self.index.d)
        tmp_index.add(candidate_vectors)
        k = min(top_k, len(candidate_indices))
        scores, local_ids = tmp_index.search(query_vector.reshape(1, -1), k)
        results = []
        for score, local_idx in zip(scores[0], local_ids[0]):
            if local_idx == -1:
                continue
            global_idx = candidate_indices[local_idx]
            results.append((self.metadata[global_idx], float(score)))
        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        # Issue 5: take the lock so concurrent upload+list doesn't race on
        # self.metadata. list_documents() is read-only so we release quickly.
        with self._lock:
            docs: Dict[str, Dict[str, Any]] = {}
            for m in self.metadata:
                doc_id = m["doc_id"]
                if doc_id not in docs:
                    docs[doc_id] = {"doc_id": doc_id, "filename": m["source"], "num_chunks": 0}
                docs[doc_id]["num_chunks"] += 1
            return list(docs.values())

    def delete_document(self, doc_id: str) -> int:
        """
        Issue 7: Remove all chunks for doc_id and rebuild the FAISS index
        from the surviving vectors.

        FAISS IndexFlatIP has no native per-vector delete. We reconstruct
        the index by extracting surviving vectors with index.reconstruct()
        and re-adding them. This is O(n) but correct; at demo/portfolio
        scale the corpus is small enough that this is fine. If the index
        ever grows large, switch the index type to IndexIDMap2 which
        supports remove_ids() natively.
        """
        with self._lock:
            keep_indices = [i for i, m in enumerate(self.metadata) if m["doc_id"] != doc_id]
            removed = len(self.metadata) - len(keep_indices)
            if removed == 0:
                raise ValueError(f"No document with doc_id '{doc_id}' found in the index.")

            # Reconstruct surviving vectors directly from the FAISS index.
            surviving_vectors = np.array(
                [self.index.reconstruct(i) for i in keep_indices], dtype="float32"
            ) if keep_indices else np.empty((0, self.index.d), dtype="float32")

            # Rebuild metadata list.
            self.metadata = [self.metadata[i] for i in keep_indices]

            # Rebuild the FAISS index.
            self.index = faiss.IndexFlatIP(self.index.d)
            if len(surviving_vectors) > 0:
                self.index.add(surviving_vectors)

            self._persist()
            return removed


# ── Singleton ──────────────────────────────────────────────────────────────
# Issue 2: guard against get_store(dim=384) silently returning a store that
# was already created with a different dim (e.g. list_documents() on startup
# vs. an upload with a different embedding model). We validate that the
# requested dim matches what's already loaded.

_store_singleton: FaissStore = None
_singleton_lock = threading.Lock()


def get_store(dim: int) -> FaissStore:
    global _store_singleton
    if _store_singleton is None:
        with _singleton_lock:
            if _store_singleton is None:       # double-checked locking
                _store_singleton = FaissStore(dim)
    # After first creation, the real dim comes from the persisted index
    # (which may differ from the hardcoded 384 passed by list_documents).
    # We only raise if a caller passes a concrete non-384 dim that doesn't
    # match — that indicates a model swap that needs a manual index rebuild.
    if dim != 384 and dim != _store_singleton.index.d:
        raise ValueError(
            f"Requested store dim {dim} does not match existing index dim "
            f"{_store_singleton.index.d}. Delete the persisted index to rebuild."
        )
    return _store_singleton
