import { SourceChunk } from "@/lib/api";

export default function SourceCard({ source, index }: { source: SourceChunk; index: number }) {
  const confidencePct = Math.round(source.score * 100);

  return (
    <div className="torn-top bg-paper-dim border border-muted/30 shadow-sm p-4 pt-5 flex flex-col gap-2 min-w-[220px] max-w-[260px]">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted">
          Source {index + 1}
        </span>
        <span className="stamp inline-block border border-rust text-rust font-mono text-[9px] uppercase px-1.5 py-0.5 rounded-sm">
          {confidencePct}% match
        </span>
      </div>
      <p className="font-body text-xs leading-relaxed text-ink/85 line-clamp-5">
        {source.text}
      </p>
      <p className="font-mono text-[10px] text-muted truncate mt-auto pt-1 border-t border-muted/20">
        {source.source} · chunk {source.chunk_index}
      </p>
    </div>
  );
}
