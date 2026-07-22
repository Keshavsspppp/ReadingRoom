from typing import List, Optional
from pydantic import BaseModel


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int


class UploadResponse(BaseModel):
    document: DocumentInfo
    message: str


class DeleteResponse(BaseModel):
    doc_id: str
    chunks_removed: int
    message: str


class SourceChunk(BaseModel):
    text: str
    source: str
    doc_id: str          # Issue 4: expose doc_id so UI can distinguish same-named files
    chunk_index: int
    score: float


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    doc_id: Optional[str] = None   # Issue 4: scope query to a single document


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    grounded: bool  # False if retrieval found nothing above min_score
