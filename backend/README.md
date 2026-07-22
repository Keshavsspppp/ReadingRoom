# RAG Document Q&A — Backend

FastAPI service implementing the full RAG pipeline: ingest → chunk → embed → store → retrieve → generate.

## Stack

| Stage        | Choice                                             |
|--------------|-----------------------------------------------------|
| Parsing      | `pypdf`, `python-docx`                              |
| Chunking     | Custom recursive character splitter (paragraph → sentence → word → hard cut), with overlap |
| Embeddings   | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector store | FAISS `IndexFlatIP` (cosine via normalized vectors), persisted to disk |
| Generation   | HuggingFace Inference API if `HF_API_TOKEN` is set, else a local `flan-t5-base` fallback |

## Setup

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit if you want hosted generation
uvicorn app.main:app --reload --port 8000
```

First request will download the embedding model (~90MB) and, if no `HF_API_TOKEN`
is set, the local generation model (~250MB) — both cached after that.

## Endpoints

- `POST /documents/upload` — multipart file upload (`.pdf`, `.docx`, `.txt`, `.md`)
- `GET /documents` — list ingested documents and chunk counts
- `POST /query` — `{"question": "..."}` → grounded answer + cited source chunks
- `GET /health` — liveness check

## Tuning knobs (all in `app/config.py`)

- `chunk_size` / `chunk_overlap` — smaller chunks improve precision on narrow
  factual questions; larger chunks preserve more context for synthesis questions.
  Worth A/B testing against a small hand-labeled Q&A set from your own documents.
- `top_k` — how many chunks get passed to the LLM as context.
- `min_score` — cosine-similarity floor below which a match is discarded as
  irrelevant (prevents the model from being fed noise and hallucinating an answer).
- `embedding_model` — swap to a larger model (e.g. `BAAI/bge-large-en-v1.5`) for
  better retrieval accuracy at the cost of slower embedding.

## Notable design decisions

- **Exact search (FAISS flat), not ANN.** At portfolio/demo scale (a handful of
  documents), exact cosine search is fast enough and removes a whole class of
  "why didn't it find the obviously relevant chunk" bugs that approximate
  indexes (IVF/HNSW) can introduce. Swapping to `IndexIVFFlat` is a one-line
  change in `vectorstore/faiss_store.py` once the corpus grows.
- **Grounding is enforced, not just prompted.** If retrieval returns nothing
  above `min_score`, the API returns a fixed "couldn't find this" message
  instead of calling the LLM at all — cheaper and removes a hallucination path.
- **HF API / local fallback split** means the whole project runs for $0 with
  zero API keys (via the local flan-t5 fallback), while still supporting a
  drop-in upgrade to much better hosted models for demo day.
