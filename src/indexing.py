from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

from .config import BM25_DIR, FAISS_DIR
from .io_utils import read_jsonl


def _safe_path(path: Path) -> str:
    """Windows'ta Unicode yol karakterlerini (ş, ü, ö vb.) FAISS C++ kütüphanesi
    için 8.3 kısa yol formatına çevirir. Diğer platformlarda düz string döner."""
    if sys.platform != "win32":
        return str(path)
    try:
        import ctypes
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
        buf = ctypes.create_unicode_buffer(1024)
        # Dizin yoksa önce oluştur, sonra kısa yolu al
        path.mkdir(parents=True, exist_ok=True)
        ret = GetShortPathNameW(str(path), buf, 1024)
        if ret and buf.value:
            return buf.value
    except Exception:
        pass
    return str(path)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_langchain_components():
    from langchain_community.docstore.document import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    return Document, HuggingFaceEmbeddings, FAISS


def _bm25_cls():
    from rank_bm25 import BM25Okapi

    return BM25Okapi


def build_indexes(chunks_path: Path) -> None:
    chunks = read_jsonl(chunks_path)
    if not chunks:
        raise RuntimeError("No chunks found. Run ingest first.")

    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "section": c.get("section"),
            "task_id": c.get("task_id"),
            "step_id": c.get("step_id"),
            "figure_ids": c.get("figure_ids", []),
            "table_ids": c.get("table_ids", []),
        }
        for c in chunks
    ]

    Document, HuggingFaceEmbeddings, FAISS = _load_langchain_components()
    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = FAISS.from_documents(docs, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(_safe_path(FAISS_DIR))

    BM25Okapi = _bm25_cls()
    tokenized = [t.lower().split() for t in texts]
    _ = BM25Okapi(tokenized)
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    (BM25_DIR / "bm25.json").write_text(
        json.dumps(
            {
                "tokenized": tokenized,
                "chunk_ids": [c["chunk_id"] for c in chunks],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_bm25() -> Dict[str, object]:
    BM25Okapi = _bm25_cls()
    path = BM25_DIR / "bm25.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    bm25 = BM25Okapi(payload["tokenized"])
    return {"bm25": bm25, "chunk_ids": payload["chunk_ids"]}
