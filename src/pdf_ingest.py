"""PDF extraction and three alternative chunking strategies."""
from __future__ import annotations
import re
import fitz
from dataclasses import dataclass, asdict
from typing import Iterable

@dataclass
class Chunk:
    id: str
    text: str
    page_start: int
    page_end: int
    section: str
    strategy: str

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_pages(pdf_path: str) -> list[dict]:
    print(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        # Blocks preserve more layout information than get_text("text").
        blocks = page.get_text("blocks", sort=True)
        text = clean_text("\n".join(b[4] for b in blocks if len(b) > 4 and isinstance(b[4], str)))
        if text:
            pages.append({"page": i + 1, "text": text})
    if not pages:
        raise ValueError("No extractable text found. The PDF may be scanned; OCR is a future extension.")
    return pages

def fixed_chunks(pages: list[dict], size: int = 1400, overlap: int = 220) -> list[Chunk]:
    """Character windows; intentionally simple baseline."""
    out, serial = [], 0
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            piece = text[start:end].strip()
            if piece:
                out.append(Chunk(f"fixed-{serial}", piece, p["page"], p["page"], "unknown", "fixed"))
                serial += 1
            if end == len(text): break
            start = max(start + 1, end - overlap)
    return out

HEADING = re.compile(
    r"(?im)^(?:(?:CHAPTER|SECTION)\s+[A-Z0-9IVX.-]+.*|"
    r"(?:[0-9]+(?:\.[0-9]+)*\.?\s+)[A-Z][A-Z0-9 ,()'’/&:-]{3,}|"
    r"[A-Z][A-Z0-9 ,()'’/&:-]{8,})$"
)


def structure_chunks(pages: list[dict], target_chars: int = 1800,
                     max_chars: int = 2600) -> list[Chunk]:
    """Split on likely headings and paragraph boundaries."""

    out = []
    serial = 0
    current_section = "Document introduction"
    buffer = ""
    pstart = None
    pend = None

    def emit():
        nonlocal serial, buffer, pstart, pend

        body = buffer.strip()

        if body:
            out.append(
                Chunk(
                    f"struct-{serial}",
                    f"SECTION CONTEXT: {current_section}\n\n{body}",
                    pstart or 1,
                    pend or pstart or 1,
                    current_section,
                    "structure"
                )
            )
            serial += 1

        buffer = ""
        pstart = None
        pend = None

    for page in pages:
        paras = [
            x.strip()
            for x in re.split(r"\n\s*\n", page["text"])
            if x.strip()
        ]

        for para in paras:
            candidate = para.replace("\n", " ").strip()

            is_heading = (
                len(candidate) < 115
                and bool(HEADING.match(candidate))
            )

            if is_heading:
                emit()
                current_section = candidate
                continue

            if buffer and len(buffer) + len(candidate) + 2 > max_chars:
                emit()

            if pstart is None:
                pstart = page["page"]

            pend = page["page"]
            buffer += candidate + "\n\n"

            if len(buffer) >= target_chars:
                emit()

    emit()
    return out


def make_chunks(pages: list[dict], strategy: str) -> list[Chunk]:
    if strategy == "fixed":
        return fixed_chunks(pages)
    if strategy == "structure":
        return structure_chunks(pages)
    raise ValueError(f"Unknown chunk strategy: {strategy}")
