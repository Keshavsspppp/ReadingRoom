from typing import List, Optional
from app.config import settings
from app.embeddings.embedder import embed_query
from app.vectorstore.faiss_store import get_store
from app.models.schemas import SourceChunk


def retrieve(
    question: str,
    top_k: Optional[int] = None,
    doc_id: Optional[str] = None,   # Issue 4: None = search whole corpus
) -> List[SourceChunk]:
    top_k = top_k or settings.top_k
    query_vec = embed_query(question)
    store = get_store(dim=query_vec.shape[0])
    hits = store.search(query_vec, top_k, doc_id=doc_id)

    results = []
    for meta, score in hits:
        if score < settings.min_score:
            continue
        results.append(
            SourceChunk(
                text=meta["text"],
                source=meta["source"],
                doc_id=meta["doc_id"],
                chunk_index=meta["chunk_index"],
                score=round(score, 4),
            )
        )
    return results
