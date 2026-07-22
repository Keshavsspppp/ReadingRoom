"use client";

import { useEffect, useState } from "react";
import { DocumentInfo, listDocuments } from "@/lib/api";
import UploadPanel from "@/components/UploadPanel";
import ChatPanel from "@/components/ChatPanel";

export default function Home() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);

  useEffect(() => {
    listDocuments().then(setDocuments).catch(() => {});
  }, []);

  return (
    <main className="min-h-screen flex flex-col">
      <header className="border-b border-muted/30 px-8 py-6">
        <p className="font-mono text-xs uppercase tracking-widest text-muted">
          Retrieval-Augmented Generation
        </p>
        <h1 className="font-display text-3xl font-bold text-ink">The Reading Room</h1>
      </header>

      <div className="flex-1 grid grid-cols-1 md:grid-cols-[340px_1fr]">
        <section className="border-b md:border-b-0 md:border-r border-muted/30 p-8">
          <UploadPanel
            documents={documents}
            onUploaded={(doc) => setDocuments((prev) => [...prev, doc])}
          />
        </section>

        <section className="p-8 flex flex-col">
          <ChatPanel hasDocuments={documents.length > 0} />
        </section>
      </div>

      <footer className="border-t border-muted/30 px-8 py-3">
        <p className="font-mono text-[10px] text-muted">
          Embeddings: sentence-transformers/all-MiniLM-L6-v2 · Index: FAISS (flat, cosine) · Chunking: recursive, 800/120
        </p>
      </footer>
    </main>
  );
}
