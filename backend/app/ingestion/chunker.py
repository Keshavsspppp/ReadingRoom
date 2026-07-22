"""
Recursive character-based chunking.

Why recursive: splitting purely on a fixed character count can cut a
sentence in half, which hurts embedding quality (the chunk no longer
represents one coherent idea). This splitter tries paragraph breaks
first, then sentence breaks, then words, only falling back to a hard
character cut as a last resort. Overlap keeps context from being lost
at chunk boundaries, which matters most for arguments that span two
paragraphs.

This is the file to point to when talking about "chunk size tuning" --
chunk_size / chunk_overlap are read from config.py so they're easy to
sweep and re-benchmark against a small eval set of Q&A pairs.
"""
from dataclasses import dataclass
from typing import List

from app.config import settings

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    text: str
    page: int
    chunk_index: int


def _split(text: str, separators: List[str], chunk_size: int) -> List[str]:
    if len(text) <= chunk_size or not separators:
        return [text]

    sep, rest_seps = separators[0], separators[1:]
    if sep == "":
        # Hard cut, last resort.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    pieces, buf = [], ""
    for part in parts:
        candidate = (buf + sep + part) if buf else part
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                pieces.append(buf)
            if len(part) > chunk_size:
                pieces.extend(_split(part, rest_seps, chunk_size))
                buf = ""
            else:
                buf = part
    if buf:
        pieces.append(buf)
    return pieces


def _add_overlap(pieces: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or len(pieces) <= 1:
        return pieces
    out = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        out.append((tail + " " + cur).strip())
    return out


def chunk_pages(pages: List[tuple], chunk_size: int = None, overlap: int = None) -> List[Chunk]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    chunks: List[Chunk] = []
    idx = 0
    for page_num, text in pages:
        text = text.strip()
        if not text:
            continue
        raw_pieces = _split(text, SEPARATORS, chunk_size)
        pieces = _add_overlap(raw_pieces, overlap)
        for piece in pieces:
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page_num, chunk_index=idx))
                idx += 1
    return chunks
