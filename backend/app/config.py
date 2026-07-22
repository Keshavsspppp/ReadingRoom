"""
Central configuration for the RAG pipeline.
All tunables (chunk size, top_k, model names) live here so they're easy
to experiment with -- this is also the file to point to when you talk
about "chunk size tuning" or "retrieval accuracy improvements".
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    # Embeddings (HuggingFace, runs locally, no API key needed)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Generation
    # If hf_api_token is set, we call the HuggingFace Inference API with
    # generation_model (better quality, needs network + a free HF token).
    # If it's empty, we fall back to a small local flan-t5 model so the
    # whole thing still runs fully offline / for free.
    hf_api_token: str = os.getenv("HF_API_TOKEN", "")
    generation_model: str = os.getenv(
        "GENERATION_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"
    )
    local_generation_model: str = os.getenv("LOCAL_GENERATION_MODEL", "google/flan-t5-base")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 800))       # characters
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 120))  # characters

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", 4))
    min_score: float = float(os.getenv("MIN_SCORE", 0.15))  # below this, treat as "no match"

    # Storage
    index_path: Path = INDEX_DIR / "faiss.index"
    metadata_path: Path = INDEX_DIR / "metadata.pkl"

    class Config:
        env_file = ".env"


settings = Settings()
