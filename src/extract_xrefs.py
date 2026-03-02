from __future__ import annotations

import re
from typing import List

from .schemas import ChunkRecord, XRefEdge

PATTERNS = [
    ("refer_to", re.compile(r"\b(?:Refer to|See)\s+([^\n\.;]+)", re.I)),
    ("task", re.compile(r"\bTASK\s+([0-9]{2}-[0-9]{2}(?:-[0-9]{2})?(?:-[A-Z0-9-]+)?)", re.I)),
    ("figure", re.compile(r"\bFigure\s+([A-Z0-9-]+)", re.I)),
    ("table", re.compile(r"\bTable\s+([A-Z0-9-]+)", re.I)),
]


def extract_xrefs(chunks: List[ChunkRecord]) -> List[XRefEdge]:
    edges: List[XRefEdge] = []
    idx = 1
    for ch in chunks:
        text = ch.text or ""
        for target_type, pattern in PATTERNS:
            for m in pattern.finditer(text):
                target = m.group(1).strip()
                edges.append(
                    XRefEdge(
                        edge_id=f"xr_{idx:06d}",
                        from_chunk_id=ch.chunk_id,
                        target_type=target_type,
                        target_key=target,
                        evidence_text=m.group(0),
                        page=ch.page_start,
                        doc_id=ch.doc_id,
                    )
                )
                idx += 1
    return edges
