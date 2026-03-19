from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import fitz

from .schemas import FigureRecord, TableRecord


def extract_figures(doc: fitz.Document, pdf_path: Path, out_dir: Path) -> List[FigureRecord]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_id = pdf_path.stem
    rows: List[FigureRecord] = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        for i, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            fig_id = f"fig_{doc_id}_p{pno+1:04d}_{i:02d}"
            img_path = out_dir / f"{fig_id}.png"
            pix.save(img_path)
            rows.append(
                FigureRecord(
                    figure_id=fig_id,
                    doc_id=doc_id,
                    page=pno + 1,
                    source_path=str(pdf_path),
                    image_path=str(img_path),
                    bbox=None,
                )
            )
    return rows


def extract_tables(pdf_path: Path) -> List[TableRecord]:
    doc_id = pdf_path.stem
    rows: List[TableRecord] = []

    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            for pno, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []
                for i, tbl in enumerate(tables, start=1):
                    if not tbl:
                        continue
                    rows.append(
                        TableRecord(
                            table_id=f"tbl_{doc_id}_p{pno:04d}_{i:02d}",
                            doc_id=doc_id,
                            page=pno,
                            source_path=str(pdf_path),
                            rows=[[c if c is not None else "" for c in r] for r in tbl],
                        )
                    )
    except Exception:
        pass

    if rows:
        return rows

    try:
        import camelot

        tables = camelot.read_pdf(str(pdf_path), pages="all")
        for i, t in enumerate(tables, start=1):
            data = t.df.fillna("").values.tolist()
            rows.append(
                TableRecord(
                    table_id=f"tbl_{doc_id}_fallback_{i:03d}",
                    doc_id=doc_id,
                    page=int(t.page),
                    source_path=str(pdf_path),
                    rows=data,
                )
            )
    except Exception:
        pass

    return rows
