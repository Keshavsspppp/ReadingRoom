import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.schemas import (
    UploadResponse, QueryRequest, QueryResponse,
    DocumentInfo, DeleteResponse,
)
from app.rag import pipeline

app = FastAPI(title="RAG Document Q&A API", version="1.0.0")

# Issue 6: lock CORS down to configured origins (default: localhost:3000 only).
# Set ALLOWED_ORIGINS=* in .env only for local dev if you need it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}

# Issue 3: max upload size in bytes, derived from MAX_UPLOAD_MB in config.
MAX_UPLOAD_BYTES = settings.max_upload_mb * 1024 * 1024


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentInfo])
def get_documents():
    return pipeline.list_documents()


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    # Issue 3: check Content-Length header before reading the body.
    # This catches the common case cheaply without streaming the whole file.
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large. Maximum upload size is {settings.max_upload_mb} MB. "
                f"Received {int(content_length) / 1024 / 1024:.1f} MB."
            ),
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    # Issue 3: enforce size limit while streaming to disk (catches cases
    # where Content-Length is missing or lying).
    bytes_written = 0
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        while chunk := await file.read(1024 * 64):   # 64 KB chunks
            bytes_written += len(chunk)
            if bytes_written > MAX_UPLOAD_BYTES:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum upload size is {settings.max_upload_mb} MB.",
                )
            tmp.write(chunk)

    try:
        doc_info = pipeline.ingest_document(tmp_path, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return UploadResponse(document=doc_info, message="Document ingested successfully.")


@app.delete("/documents/{doc_id}", response_model=DeleteResponse)
def delete_document(doc_id: str):
    """Issue 7: per-document deletion."""
    try:
        removed = pipeline.delete_document(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return DeleteResponse(
        doc_id=doc_id,
        chunks_removed=removed,
        message=f"Removed {removed} chunk(s) for document '{doc_id}'.",
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    # Issue 4: forward optional doc_id for per-document scoped queries.
    return pipeline.answer_query(request.question, request.top_k, doc_id=request.doc_id)
