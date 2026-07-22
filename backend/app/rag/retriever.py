from typing import List
from app.config import settings
from app.embeddings.embedder import embed_query
from app.vectorstore.faiss_store import get_store
from app.models.schemas import SourceChunk


def retrieve(question: str, top_k: int = None) -> List[SourceChunk]:
    top_k = top_k or settings.top_k
    query_vec = embed_query(question)
    store = get_store(dim=query_vec.shape[0])
    hits = store.search(query_vec, top_k)

    results = []
    for meta, score in hits:
        if score < settings.min_score:
            continue
        results.append(
            SourceChunk(
                text=meta["text"],
                source=meta["source"],
                chunk_index=meta["chunk_index"],
                score=round(score, 4),
            )
        )
    return results
