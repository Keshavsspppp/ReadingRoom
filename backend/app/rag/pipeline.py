import uuid
from pathlib import Path
from typing import List

from app.ingestion.loader import load_text
from app.ingestion.chunker import chunk_pages
from app.embeddings.embedder import embed_texts
from app.vectorstore.faiss_store import get_store
from app.rag.retriever import retrieve
from app.rag.generator import generate_answer
from app.models.schemas import DocumentInfo, SourceChunk, QueryResponse


def ingest_document(path: Path, filename: str) -> DocumentInfo:
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
    # dim is only needed to lazily create the store; 384 matches
    # all-MiniLM-L6-v2 and is harmless if the store already exists on disk.
    store = get_store(dim=384)
    return [DocumentInfo(**d) for d in store.list_documents()]


def answer_query(question: str, top_k: int = None) -> QueryResponse:
    chunks: List[SourceChunk] = retrieve(question, top_k)
    answer = generate_answer(question, chunks)
    return QueryResponse(answer=answer, sources=chunks, grounded=bool(chunks))
