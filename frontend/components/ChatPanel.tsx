"use client";

import { useState } from "react";
import { askQuestion, QueryResponse, DocumentInfo } from "@/lib/api";
import SourceCard from "./SourceCard";

interface Turn {
  question: string;
  response: QueryResponse;
  scopedTo?: string;   // filename of the document the question was scoped to
}

interface Props {
  hasDocuments: boolean;
  documents: DocumentInfo[];   // Issue 4: used to let user pick scope
}

export default function ChatPanel({ hasDocuments, documents }: Props) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [scopeDocId, setScopeDocId] = useState<string>("");  // "" = whole corpus

  async function handleAsk() {
    if (!question.trim() || loading) return;
    setLoading(true);
    setError(null);
    const q = question;
    const scope = scopeDocId || undefined;
    setQuestion("");
    try {
      const response = await askQuestion(q, undefined, scope);
      const scopedDoc = documents.find((d) => d.doc_id === scope);
      setTurns((prev) => [
        ...prev,
        { question: q, response, scopedTo: scopedDoc?.filename },
      ]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-6 h-full">
      <div>
        <p className="font-mono text-xs uppercase tracking-widest text-muted mb-2">
          02 — Reading desk
        </p>
        <h2 className="font-display text-2xl font-semibold text-ink">Ask the archive</h2>
      </div>

      {/* Issue 4: document scope selector */}
      {documents.length > 1 && (
        <div className="flex items-center gap-2">
          <label className="font-mono text-xs text-muted shrink-0">Scope:</label>
          <select
            value={scopeDocId}
            onChange={(e) => setScopeDocId(e.target.value)}
            className="flex-1 bg-transparent border border-muted/40 rounded-sm px-2 py-1 font-mono text-xs text-ink"
          >
            <option value="">All documents</option>
            {documents.map((doc) => (
              <option key={doc.doc_id} value={doc.doc_id}>
                {doc.filename}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="flex flex-col gap-8 flex-1 overflow-y-auto pr-1">
        {turns.length === 0 && (
          <p className="text-sm text-muted italic">
            {hasDocuments
              ? "Ask anything about the documents you've filed."
              : "File a document on the left, then ask your first question."}
          </p>
        )}

        {turns.map((turn, i) => (
          <div key={i} className="flex flex-col gap-3">
            <div className="flex items-start gap-2">
              <p className="font-display italic text-lg text-forest flex-1">
                "{turn.question}"
              </p>
              {turn.scopedTo && (
                <span className="font-mono text-[10px] text-muted border border-muted/40 rounded-sm px-1.5 py-0.5 shrink-0 mt-1">
                  {turn.scopedTo}
                </span>
              )}
            </div>
            <p className="font-body text-ink/90 leading-relaxed whitespace-pre-wrap">
              {turn.response.answer}
            </p>
            {turn.response.sources.length > 0 && (
              <div className="flex gap-3 overflow-x-auto pt-2 pb-1">
                {turn.response.sources.map((s, idx) => (
                  <SourceCard key={idx} source={s} index={idx} />
                ))}
              </div>
            )}
          </div>
        ))}

        {error && (
          <p className="font-mono text-xs text-rust border border-rust/40 rounded-sm p-2">
            {error}
          </p>
        )}
      </div>

      <div className="flex gap-2 border-t border-muted/30 pt-4">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          disabled={!hasDocuments || loading}
          placeholder={hasDocuments ? "Type your question…" : "File a document first…"}
          className="flex-1 bg-transparent border border-muted/40 focus:border-forest rounded-sm px-3 py-2 font-body text-sm text-ink placeholder:text-muted/70 disabled:opacity-50"
        />
        <button
          onClick={handleAsk}
          disabled={!hasDocuments || loading || !question.trim()}
          className="bg-forest hover:bg-forest-light disabled:opacity-40 text-paper font-body text-sm px-4 py-2 rounded-sm transition-colors"
        >
          {loading ? "Searching…" : "Ask"}
        </button>
      </div>
    </div>
  );
}
