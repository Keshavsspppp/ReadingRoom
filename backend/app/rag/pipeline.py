import uuid
from pathlib import Path
from typing import List, Optional

from app.ingestion.loader import load_text
from app.ingestion.chunker import chunk_pages
from app.embeddings.embedder import embed_texts
from app.vectorstore.faiss_store import get_store
from app.rag.retriever import retrieve
from app.rag.generator import generate_answer, _generate_locally, _build_prompt
from app.models.schemas import DocumentInfo, SourceChunk, QueryResponse


def ingest_document(path: Path, filename: str) -> DocumentInfo:
    # Issue 8: reject duplicate filenames before doing any work.
    # Two files with the same name would create separate doc_ids with
    # duplicated chunks, silently doubling retrieval noise for that content.
    store = get_store(dim=384)
    existing = {d["filename"] for d in store.list_documents()}
    if filename in existing:
        raise ValueError(
            f"A document named '{filename}' is already in the index. "
            "Delete it first or rename the file before uploading."
        )

    doc_id = uuid.uuid4().hex[:12]

    pages = load_text(path)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No extractable text found in this document.")

    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)

    metadatas = [
        {
            "doc_id": doc_id,
            "source": filename,
            "page": c.page,
            "chunk_index": c.chunk_index,
            "text": c.text,
        }
        for c in chunks
    ]

    store = get_store(dim=vectors.shape[1])
    store.add(vectors, metadatas)

    return DocumentInfo(doc_id=doc_id, filename=filename, num_chunks=len(chunks))


def list_documents() -> List[DocumentInfo]:
    store = get_store(dim=384)
    return [DocumentInfo(**d) for d in store.list_documents()]


def delete_document(doc_id: str) -> int:
    """
    Remove all chunks belonging to doc_id from the store.
    Returns the number of chunks removed. Raises ValueError if not found.

    Issue 7: FAISS IndexFlatIP has no native delete. We rebuild the index
    from the surviving metadata — O(n) but correct, and fine at demo scale.
    At larger scale, switch to IndexIDMap2 which supports remove_ids().
    """
    store = get_store(dim=384)
    return store.delete_document(doc_id)


def answer_query(
    question: str,
    top_k: Optional[int] = None,
    doc_id: Optional[str] = None,      # Issue 4: optional per-doc scoping
) -> QueryResponse:
    chunks: List[SourceChunk] = retrieve(question, top_k, doc_id=doc_id)

    # Issue 1: don't silently fall back to local — surface a clear message
    # if the hosted call fails so users know their token/model config is wrong.
    try:
        answer = generate_answer(question, chunks)
    except RuntimeError as e:
        # Hosted generation failed; use local fallback but be explicit about it.
        if chunks:
            prompt = _build_prompt(question, chunks)
            local_answer = _generate_locally(prompt)
            answer = (
                f"{local_answer}\n\n"
                f"⚠️ Hosted generation unavailable — answered with local fallback.\n"
                f"Reason: {e}"
            )
        else:
            from app.rag.generator import NO_CONTEXT_ANSWER
            answer = NO_CONTEXT_ANSWER

    return QueryResponse(answer=answer, sources=chunks, grounded=bool(chunks))
