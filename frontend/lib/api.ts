const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  num_chunks: number;
}

export interface SourceChunk {
  text: string;
  source: string;
  doc_id: string;      // Issue 4: identify which document the chunk came from
  chunk_index: number;
  score: number;
}

export interface QueryResponse {
  answer: string;
  sources: SourceChunk[];
  grounded: boolean;
}

export interface DeleteResponse {
  doc_id: string;
  chunks_removed: number;
  message: string;
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function uploadDocument(file: File): Promise<DocumentInfo> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: "POST",
    body: form,
  });
  const data = await handle<{ document: DocumentInfo; message: string }>(res);
  return data.document;
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${API_BASE}/documents`);
  return handle<DocumentInfo[]>(res);
}

// Issue 7: delete a document by its doc_id
export async function deleteDocument(docId: string): Promise<DeleteResponse> {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
  return handle<DeleteResponse>(res);
}

// Issue 4: optional docId scopes the query to a single document
export async function askQuestion(
  question: string,
  topK?: number,
  docId?: string,
): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK, doc_id: docId ?? null }),
  });
  return handle<QueryResponse>(res);
}
