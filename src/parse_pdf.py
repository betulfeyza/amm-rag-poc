from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import fitz

from .schemas import PageRecord


def parse_pdf_pages(pdf_path: Path) -> Tuple[List[PageRecord], fitz.Document]:
    doc = fitz.open(pdf_path)
    doc_id = pdf_path.stem
    pages: List[PageRecord] = []

    for pno, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        blocks = []
        for bidx, block in enumerate(page.get_text("blocks") or []):
            x1, y1, x2, y2, btext, *_ = block
            btype = _detect_block_type(btext)
            blocks.append(
                {
                    "block_id": f"{doc_id}_p{pno}_b{bidx}",
                    "type": btype,
                    "text": (btext or "").strip(),
                    "bbox": [x1, y1, x2, y2],
                }
            )
        pages.append(
            PageRecord(
                page_id=f"{doc_id}#p{pno:04d}",
                doc_id=doc_id,
                page=pno,
                source_path=str(pdf_path),
                text=text,
                layout_blocks=blocks,
                parse_confidence=1.0 if text.strip() else 0.6,
            )
        )
    return pages, doc


def _detect_block_type(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "paragraph"
    if re.search(r"\bWARNING\b", t, flags=re.I):
        return "warning"
    if re.search(r"\bCAUTION\b", t, flags=re.I):
        return "caution"
    if re.match(r"^(\d+\.|STEP\s+\d+)", t, flags=re.I):
        return "step"
    if len(t) < 80 and t.upper() == t:
        return "heading"
    return "paragraph"
