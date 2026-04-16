from __future__ import annotations

import base64
from pathlib import Path
from typing import Literal, Optional

from docx import Document
from pypdf import PdfReader


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\n".join(part for part in parts if part).strip()


def extract_docx_text(path: str) -> str:
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(parts).strip()


def extract_plain_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore").strip()


def encode_image_to_data_url(path: str, mime_type: str) -> str:
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def detect_kind(filename: str) -> Literal["image", "document", "unknown"]:
    lower = filename.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return "image"
    if lower.endswith((".pdf", ".docx", ".txt", ".md")):
        return "document"
    return "unknown"


def extract_document_text(path: str, filename: str) -> Optional[str]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_pdf_text(path)
    if lower.endswith(".docx"):
        return extract_docx_text(path)
    if lower.endswith((".txt", ".md")):
        return extract_plain_text(path)
    return None


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
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