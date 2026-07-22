from typing import List, Optional
from pydantic import BaseModel


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int


class UploadResponse(BaseModel):
    document: DocumentInfo
    message: str


class SourceChunk(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    grounded: bool  # False if retrieval found nothing above min_score
