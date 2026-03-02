from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PageRecord:
    page_id: str
    doc_id: str
    page: int
    source_path: str
    text: str
    layout_blocks: List[Dict[str, Any]] = field(default_factory=list)
    parse_confidence: float = 1.0


@dataclass
class TableRecord:
    table_id: str
    doc_id: str
    page: int
    source_path: str
    rows: List[List[str]]


@dataclass
class FigureRecord:
    figure_id: str
    doc_id: str
    page: int
    source_path: str
    image_path: str
    bbox: Optional[List[float]] = None


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    text: str
    section: Optional[str] = None
    task_id: Optional[str] = None
    step_id: Optional[str] = None
    figure_ids: List[str] = field(default_factory=list)
    table_ids: List[str] = field(default_factory=list)
    xref_targets: List[str] = field(default_factory=list)


@dataclass
class XRefEdge:
    edge_id: str
    from_chunk_id: str
    target_type: str
    target_key: str
    evidence_text: str
    page: int
    doc_id: str


@dataclass
class Citation:
    doc_id: str
    page: int
    section: Optional[str] = None
    task_id: Optional[str] = None
    step_id: Optional[str] = None


@dataclass
class EvidenceItem:
    chunk_id: str
    score: float
    text: str
    figure_ids: List[str]
    table_ids: List[str]


@dataclass
class AnswerPacket:
    answer: str
    abstained: bool
    citations: List[Citation]
    evidence: List[EvidenceItem]

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["citations"] = [asdict(c) for c in self.citations]
        out["evidence"] = [asdict(e) for e in self.evidence]
        return out
