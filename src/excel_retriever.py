from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9\-_/\.]+", (text or "").lower())


def _parse_page(value) -> int:
    if value is None:
        return 0
    m = re.search(r"(\d{1,4})", str(value))
    return int(m.group(1)) if m else 0


def _parse_page_from_source(source_text: str, sheet_name: str = "") -> int:
    src = source_text or ""
    m = re.search(r"\bPage\s+(\d{1,4})\b", src, re.I)
    if m:
        return int(m.group(1))
    s = sheet_name or ""
    m2 = re.search(r"\bPage\s+(\d{1,4})\b", s, re.I)
    if m2:
        return int(m2.group(1))
    return 0


def _parse_doc_id(source_text: str) -> str | None:
    src = source_text or ""
    task_m = re.search(r"\b([0-9]{2}-[0-9]{2}-[0-9]{2})\b", src)
    if task_m:
        return task_m.group(1)
    return None


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _extract_task_section_step(text: str) -> tuple[str | None, str | None, str | None]:
    task_m = re.search(r"\b(?:SUBTASK\s+)?([0-9]{2}-[0-9]{2}-[0-9]{2}(?:-[0-9]{3}(?:-[0-9]{3})?)?)\b", text, re.I)
    section_m = re.search(r"\b([0-9]{2}-[0-9]{2}-[0-9]{2})\b", text)
    step_m = re.search(r"\bSTEP\s+(\d+)\b", text, re.I)
    if not step_m:
        step_m = re.search(r"^\s*\((\d+)\)", text)
    task_id = task_m.group(1) if task_m else None
    section = section_m.group(1) if section_m else None
    step_id = step_m.group(1) if step_m else None
    return task_id, section, step_id


def _extract_task_code(text: str) -> str | None:
    m = re.search(r"\b([0-9]{2}-[0-9]{2}-[0-9]{2}(?:-[0-9]{3}(?:-[0-9]{3})?)?)\b", text or "")
    return m.group(1) if m else None


def _load_rows(excel_path: Path, sheet_name: str | None) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel bulunamadi: {excel_path}")

    if sheet_name:
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        df["__sheet_name"] = sheet_name
        return df

    sheets = pd.read_excel(excel_path, sheet_name=None)
    frames = []
    for name, df in sheets.items():
        if name == "Camelot_Tablolar":
            continue
        df = df.copy()
        df["__sheet_name"] = name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def retrieve_from_excel(
    query: str,
    excel_path: Path,
    topk: int = 8,
    sheet_name: str | None = None,
    exclude_header_footer: bool = True,
    min_score: float = 0.2,
    graph_hops: int = 1,
) -> List[Dict]:
    df = _load_rows(excel_path, sheet_name)
    if df.empty:
        return []

    content_col = _pick_col(df, ["İçerik", "Content"])
    source_pdf_col = _pick_col(df, ["Kaynak PDF", "Source PDF", "Source"])
    page_col = _pick_col(df, ["Sayfa (Footer)", "Page", "Footer Page"])
    id_col = _pick_col(df, ["ID", "Id", "id"])
    type_col = _pick_col(df, ["Tespit Edilen Tür", "Type"])

    if not content_col:
        raise ValueError("Excel kolonu eksik: İçerik/Content")

    if not source_pdf_col and not page_col:
        raise ValueError("Excel kolonu eksik: Kaynak PDF/Source ve Sayfa bilgisi")

    if exclude_header_footer and type_col:
        tvals = df[type_col].fillna("").astype(str)
        df = df[~tvals.str.contains("header/footer|page header", case=False, regex=True)]

    df = df[df[content_col].fillna("").astype(str).str.strip() != ""].copy()
    if df.empty:
        return []

    corpus = [str(t) for t in df[content_col].tolist()]
    tokenized = [_tokenize(t) for t in corpus]
    if not tokenized:
        return []

    bm25 = BM25Okapi(tokenized)
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)
    if len(scores) == 0:
        return []

    max_score = max(float(s) for s in scores) if len(scores) else 0.0
    if max_score <= 0:
        return []

    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    query_task = _extract_task_code(query)

    candidates: List[Dict] = []
    query_token_set = set(q_tokens)
    for i in ranked_idx:
        raw_score = float(scores[i])
        if raw_score <= 0:
            continue
        row = df.iloc[i]
        text = str(row.get(content_col, ""))
        task_id, section, step_id = _extract_task_section_step(text)
        source_val = str(row.get(source_pdf_col, "")) if source_pdf_col else ""
        if source_pdf_col and (".PDF" in source_val.upper() or ".pdf" in source_val):
            doc_id = source_val.replace(".PDF", "").replace(".pdf", "")
        else:
            doc_id = _parse_doc_id(source_val) or "excel_manual"

        if page_col:
            page_start = _parse_page(row.get(page_col))
        else:
            page_start = _parse_page_from_source(source_val, str(row.get("__sheet_name", "")))

        chunk_id = str(row.get(id_col) if id_col else "") or f"excel_{i + 1:04d}"
        norm_score = raw_score / max_score

        row_task = task_id or _extract_task_code(source_val) or _extract_task_code(text)
        task_match = bool(query_task and row_task and row_task == query_task)

        text_tokens = set(_tokenize(text))
        candidates.append(
            {
                "chunk_id": chunk_id,
                "score": norm_score,
                "text": text,
                "doc_id": doc_id,
                "page_start": page_start,
                "section": section,
                "task_id": task_id,
                "step_id": step_id,
                "figure_ids": [],
                "table_ids": [],
                "_task_match": task_match,
                "_query_overlap": len(text_tokens.intersection(query_token_set)),
                "_rank": i,
            }
        )

    if not candidates:
        return []

    if query_task:
        matched = [c for c in candidates if c.get("_task_match")]
        if matched:
            candidates = matched

    filtered = [c for c in candidates if float(c.get("score", 0.0)) >= min_score]
    if not filtered:
        filtered = candidates[:1]

    selected: List[Dict] = []
    seen_ids: set[str] = set()

    def add_item(item: Dict) -> None:
        cid = str(item.get("chunk_id", ""))
        if not cid or cid in seen_ids:
            return
        seen_ids.add(cid)
        selected.append(item)

    seeds = sorted(
        filtered,
        key=lambda x: (
            -int(bool(x.get("_task_match"))),
            -float(x.get("_query_overlap", 0)),
            -float(x.get("score", 0.0)),
            int(x.get("_rank", 0)),
        ),
    )
    for s in seeds:
        add_item(s)
        if len(selected) >= topk:
            break

    if graph_hops > 0 and len(selected) < topk:
        by_id = {c["chunk_id"]: c for c in candidates}
        by_task: Dict[str, List[str]] = {}
        by_doc_page: Dict[tuple[str, int], List[str]] = {}
        by_doc_section: Dict[tuple[str, str], List[str]] = {}

        for c in candidates:
            cid = c["chunk_id"]
            task = (c.get("task_id") or "").strip()
            doc = (c.get("doc_id") or "").strip()
            page = int(c.get("page_start") or 0)
            section = (c.get("section") or "").strip()
            if task:
                by_task.setdefault(task, []).append(cid)
            if doc and page:
                by_doc_page.setdefault((doc, page), []).append(cid)
            if doc and section:
                by_doc_section.setdefault((doc, section), []).append(cid)

        def neighbors(cid: str) -> List[str]:
            c = by_id.get(cid)
            if not c:
                return []
            doc = (c.get("doc_id") or "").strip()
            page = int(c.get("page_start") or 0)
            task = (c.get("task_id") or "").strip()
            section = (c.get("section") or "").strip()
            neigh: List[str] = []
            if task:
                neigh.extend(by_task.get(task, []))
            if doc and page:
                neigh.extend(by_doc_page.get((doc, page), []))
                neigh.extend(by_doc_page.get((doc, page - 1), []))
                neigh.extend(by_doc_page.get((doc, page + 1), []))
            if doc and section:
                neigh.extend(by_doc_section.get((doc, section), []))
            out: List[str] = []
            local_seen: set[str] = set()
            for n in neigh:
                if n != cid and n not in local_seen:
                    local_seen.add(n)
                    out.append(n)
            return out

        queue = deque()
        depth_seen: Dict[str, int] = {}
        for s in seeds[: min(3, len(seeds))]:
            sid = s["chunk_id"]
            queue.append(sid)
            depth_seen[sid] = 0

        while queue and len(selected) < topk:
            cur = queue.popleft()
            cur_depth = depth_seen.get(cur, 0)
            if cur_depth >= graph_hops:
                continue

            neighs = neighbors(cur)
            neigh_objs = [by_id[n] for n in neighs if n in by_id and n not in seen_ids]
            neigh_objs.sort(
                key=lambda x: (
                    -int(bool(x.get("_task_match"))),
                    -float(x.get("_query_overlap", 0)),
                    -float(x.get("score", 0.0)),
                    int(x.get("_rank", 0)),
                )
            )

            for nobj in neigh_objs:
                add_item(nobj)
                nid = nobj["chunk_id"]
                if nid not in depth_seen:
                    depth_seen[nid] = cur_depth + 1
                    queue.append(nid)
                if len(selected) >= topk:
                    break

    rows: List[Dict] = []
    for c in selected[:topk]:
        c.pop("_task_match", None)
        c.pop("_query_overlap", None)
        c.pop("_rank", None)
        rows.append(c)

    return rows
