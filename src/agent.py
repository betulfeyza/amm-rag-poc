from __future__ import annotations

from typing import Dict, List

from .schemas import AnswerPacket, Citation, EvidenceItem


def build_answer(query: str, evidence_rows: List[Dict]) -> AnswerPacket:
    if not evidence_rows:
        return AnswerPacket(
            answer="Sonuc: AMM icinde dogrulanamadi.",
            abstained=True,
            citations=[],
            evidence=[],
        )

    top = evidence_rows[0]
    citations = []
    evidence = []
    for row in evidence_rows:
        citations.append(
            Citation(
                doc_id=row["doc_id"],
                page=row["page_start"],
                section=row.get("section"),
                task_id=row.get("task_id"),
                step_id=row.get("step_id"),
            )
        )
        evidence.append(
            EvidenceItem(
                chunk_id=row["chunk_id"],
                score=float(row["score"]),
                text=row["text"][:1200],
                figure_ids=row.get("figure_ids", []),
                table_ids=row.get("table_ids", []),
            )
        )

    # Extractive deterministic response grounded only in top evidence.
    snippet = " ".join(top["text"].split())
    snippet = snippet[:600]
    answer = (
        f"AMM bulgusu: {snippet}\n"
        f"Kaynak: [{top['doc_id']} p.{top['page_start']} task={top.get('task_id') or '-'} step={top.get('step_id') or '-'}]"
    )

    return AnswerPacket(answer=answer, abstained=False, citations=citations, evidence=evidence)
