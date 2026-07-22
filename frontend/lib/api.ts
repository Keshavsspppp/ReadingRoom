const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  num_chunks: number;
}

export interface SourceChunk {
  text: string;
  source: string;
  chunk_index: number;
  score: number;
}

export interface QueryResponse {
  answer: string;
  sources: SourceChunk[];
  grounded: boolean;
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

export async function askQuestion(question: string, topK?: number): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
  });
  return handle<QueryResponse>(res);
}
