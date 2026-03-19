from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .chunking import build_chunks
from .config import DEFAULT_PDFS, FIGURES_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs
from .extract_tables_figures import extract_figures, extract_tables
from .extract_xrefs import extract_xrefs
from .indexing import build_indexes
from .io_utils import write_jsonl
from .parse_pdf import parse_pdf_pages


def run_ingest(pdf_paths: Iterable[Path] | None = None) -> dict:
    ensure_dirs()
    selected = [Path(p) for p in (pdf_paths or DEFAULT_PDFS)]

    pages_all = []
    tables_all = []
    figures_all = []
    chunks_all = []
    xrefs_all = []

    for src in selected:
        if not src.exists():
            raise FileNotFoundError(f"PDF not found: {src}")

        dst = RAW_DIR / src.name
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())

        pages, doc = parse_pdf_pages(dst)
        tables = extract_tables(dst)
        figures = extract_figures(doc, dst, FIGURES_DIR)
        chunks = build_chunks(pages, tables, figures)
        xrefs = extract_xrefs(chunks)

        pages_all.extend(pages)
        tables_all.extend(tables)
        figures_all.extend(figures)
        chunks_all.extend(chunks)
        xrefs_all.extend(xrefs)

        doc.close()

    write_jsonl(PROCESSED_DIR / "pages.jsonl", pages_all)
    write_jsonl(PROCESSED_DIR / "tables.jsonl", tables_all)
    write_jsonl(PROCESSED_DIR / "figures.jsonl", figures_all)
    write_jsonl(PROCESSED_DIR / "chunks.jsonl", chunks_all)
    write_jsonl(PROCESSED_DIR / "xrefs.jsonl", xrefs_all)

    build_indexes(PROCESSED_DIR / "chunks.jsonl")

    return {
        "pdfs": [str(p) for p in selected],
        "pages": len(pages_all),
        "tables": len(tables_all),
        "figures": len(figures_all),
        "chunks": len(chunks_all),
        "xrefs": len(xrefs_all),
    }
