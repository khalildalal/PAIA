"""
Document and file processing helpers

This file contains utility functions that help the app work with uploaded files.

What this file does:
1. Extract text from PDF files
2. Extract text from Word (.docx) files
3. Read plain text files
4. Convert images into data URLs for vision-based solving
5. Detect whether an uploaded file is an image or a document
6. Split long extracted text into smaller chunks for RAG indexing

Why this matters:
The main app should stay focused on routes and logic.
These helpers keep file-processing tasks organized in one place.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal, Optional

from docx import Document
from pypdf import PdfReader


def extract_pdf_text(path: str) -> str:
    """Read all pages of a PDF and combine their extracted text into one string."""
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join(part for part in parts if part).strip()


def extract_docx_text(path: str) -> str:
    """Read paragraph text from a Word document."""
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(parts).strip()


def extract_plain_text(path: str) -> str:
    """Read a plain text or markdown file as text."""
    return Path(path).read_text(encoding="utf-8", errors="ignore").strip()


def encode_image_to_data_url(path: str, mime_type: str) -> str:
    """Convert an image file into a base64 data URL for API input."""
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def detect_kind(filename: str) -> Literal["image", "document", "unknown"]:
    """Classify a filename as image, document, or unknown based on its extension."""
    lower = filename.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return "image"
    if lower.endswith((".pdf", ".docx", ".txt", ".md")):
        return "document"
    return "unknown"


def extract_document_text(path: str, filename: str) -> Optional[str]:
    """Choose the correct extraction method based on file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_pdf_text(path)
    if lower.endswith(".docx"):
        return extract_docx_text(path)
    if lower.endswith((".txt", ".md")):
        return extract_plain_text(path)
    return None


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into cleaner single-space text."""
    return " ".join((text or "").split()).strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split long text into overlapping chunks for retrieval and embeddings."""
    clean = normalize_whitespace(text)
    if not clean:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks: list[str] = []
    start = 0
    length = len(clean)

    while start < length:
        end = min(length, start + chunk_size)
        chunk = clean[start:end].strip()

        if end < length:
            last_space = chunk.rfind(" ")
            if last_space > int(chunk_size * 0.6):
                end = start + last_space
                chunk = clean[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        start = max(0, end - overlap)

    return chunks