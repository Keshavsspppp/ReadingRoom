import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import UploadResponse, QueryRequest, QueryResponse, DocumentInfo
from app.rag import pipeline

app = FastAPI(title="RAG Document Q&A API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before deploying publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentInfo])
def get_documents():
    return pipeline.list_documents()


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_SUFFIXES)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        doc_info = pipeline.ingest_document(tmp_path, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return UploadResponse(document=doc_info, message="Document ingested successfully.")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return pipeline.answer_query(request.question, request.top_k)
