"""
Turns an uploaded file into plain text, page by page where possible.
Supports PDF, DOCX and plain text -- add new formats by adding a
branch in `load_text`.
"""
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader
import docx


def _load_pdf(path: Path) -> List[Tuple[int, str]]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def _load_docx(path: Path) -> List[Tuple[int, str]]:
    document = docx.Document(str(path))
    text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [(1, text)] if text.strip() else []


def _load_txt(path: Path) -> List[Tuple[int, str]]:
    text = path.read_text(errors="ignore")
    return [(1, text)] if text.strip() else []


def load_text(path: Path) -> List[Tuple[int, str]]:
    """
    Returns a list of (page_number, text) tuples. Page number is 1 for
    formats without a native concept of pages (docx, txt).
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix in (".txt", ".md"):
        return _load_txt(path)
    raise ValueError(f"Unsupported file type: {suffix}")
