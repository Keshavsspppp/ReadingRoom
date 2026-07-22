# The Reading Room — Frontend

Next.js 14 (App Router, TypeScript, Tailwind) UI for the RAG backend.

Two-pane "desk" layout: an intake desk on the left for uploading documents,
a reading desk on the right for asking questions. Retrieved chunks are shown
as torn-edge index cards with a rubber-stamped confidence score, so it's
visible *why* the model answered the way it did — the actual passages backing
each answer, not just a black-box response.

## Setup

```bash
cd frontend
npm install
cp .env.local.example .env.local   # point at your backend URL
npm run dev
```

Runs at `http://localhost:3000`, expects the backend at
`http://localhost:8000` by default (`NEXT_PUBLIC_API_URL`).

## Structure

```
app/page.tsx              — layout, wires upload + chat state together
components/UploadPanel.tsx — drag/drop upload + card catalog of ingested docs
components/ChatPanel.tsx   — question input, answer stream, source cards
components/SourceCard.tsx  — the torn-index-card citation component
lib/api.ts                 — typed fetch wrapper for the backend API
```
