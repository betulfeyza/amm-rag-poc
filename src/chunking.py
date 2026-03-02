from __future__ import annotations

import re
from typing import Dict, List

from .config import CHUNK_MAX_CHARS
from .schemas import ChunkRecord, FigureRecord, PageRecord, TableRecord


def build_chunks(
    pages: List[PageRecord],
    tables: List[TableRecord],
    figures: List[FigureRecord],
) -> List[ChunkRecord]:
    chunks: List[ChunkRecord] = []
    tables_by_page: Dict[int, List[str]] = {}
    figs_by_page: Dict[int, List[str]] = {}

    for t in tables:
        tables_by_page.setdefault(t.page, []).append(t.table_id)
    for f in figures:
        figs_by_page.setdefault(f.page, []).append(f.figure_id)

    cur_text = []
    cur_start = 1
    chunk_idx = 1
    cur_task = None
    cur_section = None
    cur_step = None

    def flush(end_page: int) -> None:
        nonlocal cur_text, cur_start, chunk_idx, cur_task, cur_section, cur_step
        text = "\n".join(cur_text).strip()
        if not text:
            return
        chunk_id = f"ch_{pages[0].doc_id}_{chunk_idx:05d}"
        fig_ids = []
        tbl_ids = []
        for p in range(cur_start, end_page + 1):
            fig_ids.extend(figs_by_page.get(p, []))
            tbl_ids.extend(tables_by_page.get(p, []))
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=pages[0].doc_id,
                page_start=cur_start,
                page_end=end_page,
                text=text,
                section=cur_section,
                task_id=cur_task,
                step_id=cur_step,
                figure_ids=sorted(set(fig_ids)),
                table_ids=sorted(set(tbl_ids)),
                xref_targets=[],
            )
        )
        chunk_idx += 1
        cur_text = []

    for pg in pages:
        text = pg.text or ""
        task_m = re.search(r"\bTASK\s+([0-9]{2}-[0-9]{2}(?:-[0-9]{2})?(?:-[A-Z0-9-]+)?)", text, re.I)
        if task_m:
            cur_task = task_m.group(1)
        sec_m = re.search(r"\b([0-9]{2}-[0-9]{2}-[0-9]{2})\b", text)
        if sec_m:
            cur_section = sec_m.group(1)
        step_m = re.search(r"\b(?:STEP\s+|^)(\d+)\b", text, re.I | re.M)
        if step_m:
            cur_step = step_m.group(1)

        for para in split_paragraphs(text):
            projected = ("\n".join(cur_text) + "\n" + para).strip()
            if len(projected) > CHUNK_MAX_CHARS and cur_text:
                flush(pg.page)
                cur_start = pg.page
            cur_text.append(para)

    if pages:
        flush(pages[-1].page)
    return chunks


def split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text or "")]
    return [p for p in parts if p]
