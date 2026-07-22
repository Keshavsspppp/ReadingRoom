"use client";

import { useCallback, useRef, useState } from "react";
import { DocumentInfo, uploadDocument } from "@/lib/api";

interface Props {
  documents: DocumentInfo[];
  onUploaded: (doc: DocumentInfo) => void;
}

export default function UploadPanel({ documents, onUploaded }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setUploading(true);
      setError(null);
      try {
        const doc = await uploadDocument(file);
        onUploaded(doc);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Upload failed.");
      } finally {
        setUploading(false);
      }
    },
    [onUploaded]
  );

  return (
    <div className="flex flex-col gap-6">
      <div>
        <p className="font-mono text-xs uppercase tracking-widest text-muted mb-2">
          01 — Intake
        </p>
        <h2 className="font-display text-2xl font-semibold text-ink">Bring a document</h2>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const file = e.dataTransfer.files?.[0];
          if (file) handleFile(file);
        }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-sm border-2 border-dashed p-8 text-center transition-colors ${
          dragOver ? "border-rust bg-paper-dim" : "border-muted/50 hover:border-forest"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.docx,.txt,.md"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
        <p className="font-body text-sm text-ink/80">
          {uploading ? "Filing document into the index…" : "Drop a PDF, DOCX, or TXT here"}
        </p>
        <p className="font-mono text-xs text-muted mt-1">or click to browse</p>
      </div>

      {error && (
        <p className="font-mono text-xs text-rust border border-rust/40 rounded-sm p-2">
          {error}
        </p>
      )}

      <div>
        <p className="font-mono text-xs uppercase tracking-widest text-muted mb-3">
          Card catalog ({documents.length})
        </p>
        {documents.length === 0 ? (
          <p className="text-sm text-muted italic">Nothing filed yet.</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {documents.map((doc) => (
              <li
                key={doc.doc_id}
                className="flex items-center justify-between border-b border-muted/30 pb-2"
              >
                <span className="font-body text-sm text-ink truncate pr-2">{doc.filename}</span>
                <span className="font-mono text-xs text-muted shrink-0">
                  {doc.num_chunks} chunks
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
