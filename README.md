# RAG Document Q&A — "The Reading Room"

Upload documents, ask questions, get answers grounded in your own material
with the exact source passages cited.

```
Upload → Extract text → Chunk (recursive, w/ overlap) → Embed (MiniLM)
       → Store (FAISS) → [question] → Retrieve top-k → Generate (HF LLM)
```

## Repo layout

```
backend/    FastAPI service: ingestion, chunking, embeddings, FAISS, generation
frontend/   Next.js UI: upload panel + chat panel with cited source cards
```

Each half has its own README with setup instructions:
[`backend/README.md`](backend/README.md) · [`frontend/README.md`](frontend/README.md)

## Quickstart

```bash
# Terminal 1
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`, drop in a PDF, and start asking questions.

## What's implemented

- [x] Multi-format ingestion (PDF, DOCX, TXT/MD)
- [x] Recursive chunking with overlap (tunable chunk size)
- [x] Local, free embeddings (sentence-transformers)
- [x] FAISS vector store with disk persistence
- [x] Retrieval with a similarity floor (won't force an answer with no evidence)
- [x] Generation via HF Inference API or a fully local fallback (zero API keys needed)
- [x] UI showing exactly which passages back each answer, with a match score

## Where to take it next

- Swap `IndexFlatIP` for `IndexIVFFlat`/HNSW once the corpus gets large
- Add per-document deletion (currently metadata is append-only)
- Add a small labeled eval set (question → expected doc/chunk) to actually
  measure retrieval accuracy when tuning chunk size / top_k, rather than
  eyeballing it
- Streaming responses (SSE) instead of waiting for the full generation
- Deploy: backend to Render/Fly.io, frontend to Vercel
