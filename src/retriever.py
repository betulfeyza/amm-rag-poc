from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .config import DENSE_WEIGHT, FAISS_DIR, MIN_SCORE, PROCESSED_DIR, SPARSE_WEIGHT
from .indexing import load_bm25
from .io_utils import read_jsonl


def _safe_path(path) -> str:
    import sys
    if sys.platform != "win32":
        return str(path)
    try:
        import ctypes
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
        buf = ctypes.create_unicode_buffer(1024)
        ret = GetShortPathNameW(str(path), buf, 1024)
        if ret and buf.value:
            return buf.value
    except Exception:
        pass
    return str(path)


def _load_faiss():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(_safe_path(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)


def retrieve(query: str, topk: int = 8) -> List[Dict]:
    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    xrefs_path = PROCESSED_DIR / "xrefs.jsonl"

    if not chunks_path.exists() or not (FAISS_DIR / "index.faiss").exists():
        return []

    chunks = read_jsonl(chunks_path)
    if not chunks:
        return []

    xrefs = read_jsonl(xrefs_path)
    by_id = {c["chunk_id"]: c for c in chunks}

    dense_scores = _dense_scores(query, topk * 4)
    sparse_scores = _sparse_scores(query, topk * 4)

    fused = defaultdict(float)
    for cid, score in dense_scores.items():
        fused[cid] += DENSE_WEIGHT * score
    for cid, score in sparse_scores.items():
        fused[cid] += SPARSE_WEIGHT * score

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    base = [cid for cid, score in ranked if score >= MIN_SCORE][:topk]
    expanded = expand_with_xrefs(base, chunks, xrefs, limit=max(2, topk // 3))

    final_ids: List[str] = []
    for cid in base + expanded:
        if cid in by_id and cid not in final_ids:
            final_ids.append(cid)
    final_ids = final_ids[:topk]

    return [
        {
            "chunk_id": cid,
            "score": float(fused.get(cid, 0.0)),
            "text": by_id[cid]["text"],
            "doc_id": by_id[cid]["doc_id"],
            "page_start": by_id[cid]["page_start"],
            "section": by_id[cid].get("section"),
            "task_id": by_id[cid].get("task_id"),
            "step_id": by_id[cid].get("step_id"),
            "figure_ids": by_id[cid].get("figure_ids", []),
            "table_ids": by_id[cid].get("table_ids", []),
        }
        for cid in final_ids
    ]


def _dense_scores(query: str, k: int) -> Dict[str, float]:
    store = _load_faiss()
    out: Dict[str, float] = {}
    docs = store.similarity_search_with_score(query, k=k)
    if not docs:
        return out

    raw = [score for _, score in docs]
    min_s, max_s = min(raw), max(raw)
    for doc, score in docs:
        cid = doc.metadata["chunk_id"]
        if max_s == min_s:
            norm = 1.0
        else:
            norm = 1.0 - ((score - min_s) / (max_s - min_s))
        out[cid] = max(out.get(cid, 0.0), float(norm))
    return out


def _sparse_scores(query: str, k: int) -> Dict[str, float]:
    bm25_path = Path("data/index/bm25/bm25.json")
    if not bm25_path.exists():
        return {}

    data = load_bm25()
    bm25 = data["bm25"]
    chunk_ids = data["chunk_ids"]
    scores = bm25.get_scores(query.lower().split())
    idx_scores = list(enumerate(scores))
    idx_scores.sort(key=lambda x: x[1], reverse=True)
    top = idx_scores[:k]
    if not top:
        return {}
    max_s = max([s for _, s in top]) or 1.0
    return {chunk_ids[i]: float(s / max_s) for i, s in top}


def expand_with_xrefs(base_ids: List[str], chunks: List[Dict], xrefs: List[Dict], limit: int = 2) -> List[str]:
    by_id = {c["chunk_id"]: c for c in chunks}
    found: List[str] = []

    for edge in xrefs:
        if edge["from_chunk_id"] not in base_ids:
            continue
        key = edge["target_key"].lower()
        for c in chunks:
            if c["chunk_id"] in base_ids or c["chunk_id"] in found:
                continue
            hay = " ".join(
                [
                    c.get("task_id") or "",
                    c.get("section") or "",
                    " ".join(c.get("figure_ids", [])),
                    " ".join(c.get("table_ids", [])),
                    c.get("text", "")[:300],
                ]
            ).lower()
            if key and key in hay:
                found.append(c["chunk_id"])
                if len(found) >= limit:
                    return found

    for cid in base_ids:
        if cid in by_id and len(found) < limit:
            siblings = [
                c["chunk_id"]
                for c in chunks
                if c["doc_id"] == by_id[cid]["doc_id"]
                and abs(c["page_start"] - by_id[cid]["page_start"]) <= 1
                and c["chunk_id"] not in base_ids
                and c["chunk_id"] not in found
            ]
            for s in siblings:
                found.append(s)
                if len(found) >= limit:
                    return found
    return found
