from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .config import (
    BASE_DIR,
    GEMINI_DEFAULT_MODEL,
    GEMINI_TEMPERATURE,
    OPENAI_DEFAULT_MODEL,
    OPENAI_MAX_EVIDENCE,
    OPENAI_TEMPERATURE,
)
from .schemas import AnswerPacket, Citation, EvidenceItem


def _load_prompt(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback


def _build_citations_and_evidence(evidence_rows: List[Dict]) -> tuple[List[Citation], List[EvidenceItem]]:
    citations: List[Citation] = []
    evidence: List[EvidenceItem] = []
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
    return citations, evidence


def _abstain_packet() -> AnswerPacket:
    return AnswerPacket(
        answer="Sonuc: AMM icinde dogrulanamadi.",
        abstained=True,
        citations=[],
        evidence=[],
    )


def build_answer(query: str, evidence_rows: List[Dict]) -> AnswerPacket:
    if not evidence_rows:
        return _abstain_packet()

    top = evidence_rows[0]
    citations, evidence = _build_citations_and_evidence(evidence_rows)

    snippet = " ".join(top["text"].split())
    snippet = snippet[:600]
    answer = (
        f"AMM bulgusu: {snippet}\n"
        f"Kaynak: [{top['doc_id']} p.{top['page_start']} task={top.get('task_id') or '-'} step={top.get('step_id') or '-'}]"
    )

    return AnswerPacket(answer=answer, abstained=False, citations=citations, evidence=evidence)


def build_answer_openai(
    query: str,
    evidence_rows: List[Dict],
    model: str = OPENAI_DEFAULT_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
) -> AnswerPacket:
    if not evidence_rows:
        return _abstain_packet()

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI modu icin 'langchain-openai' paketi gerekli. pip install -r requirements.txt"
        ) from exc

    system_prompt = _load_prompt(
        BASE_DIR / "prompts" / "system_answering.txt",
        "You are an AMM-grounded assistant. Use only provided evidence and cite sources.",
    )
    abstain_policy = _load_prompt(
        BASE_DIR / "prompts" / "abstain_policy.txt",
        "If evidence is insufficient, abstain and do not guess.",
    )

    top_rows = evidence_rows[:OPENAI_MAX_EVIDENCE]
    evidence_blocks: List[str] = []
    for idx, row in enumerate(top_rows, start=1):
        evidence_blocks.append(
            "\n".join(
                [
                    f"[{idx}] doc={row['doc_id']} page={row['page_start']} section={row.get('section') or '-'} task={row.get('task_id') or '-'} step={row.get('step_id') or '-'} score={float(row['score']):.3f}",
                    row["text"][:2000],
                ]
            )
        )

    human_prompt = "\n\n".join(
        [
            f"Soru: {query}",
            "Asagidaki AMM kanitlarini kullanarak kisa ve net cevap ver.",
            "Her iddia kanitta gecmiyorsa o iddiayi verme.",
            "Mutlaka kaynak satiri ekle: [doc_id p.page task step]",
            abstain_policy,
            "KANITLAR:",
            "\n\n".join(evidence_blocks),
        ]
    )

    try:
        llm = ChatOpenAI(model=model, temperature=temperature)
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    except Exception as exc:
        raise RuntimeError(f"OpenAI cagrisi basarisiz: {exc}") from exc

    answer_text = (response.content or "").strip()
    if not answer_text:
        return build_answer(query, evidence_rows)

    citations, evidence = _build_citations_and_evidence(evidence_rows)
    return AnswerPacket(answer=answer_text, abstained=False, citations=citations, evidence=evidence)


def build_answer_gemini(
    query: str,
    evidence_rows: List[Dict],
    model: str = GEMINI_DEFAULT_MODEL,
    temperature: float = GEMINI_TEMPERATURE,
) -> AnswerPacket:
    if not evidence_rows:
        return _abstain_packet()

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "Gemini modu icin 'langchain-google-genai' paketi gerekli. pip install -r requirements.txt"
        ) from exc

    system_prompt = _load_prompt(
        BASE_DIR / "prompts" / "system_answering.txt",
        "You are an AMM-grounded assistant. Use only provided evidence and cite sources.",
    )
    abstain_policy = _load_prompt(
        BASE_DIR / "prompts" / "abstain_policy.txt",
        "If evidence is insufficient, abstain and do not guess.",
    )

    top_rows = evidence_rows[:OPENAI_MAX_EVIDENCE]
    evidence_blocks: List[str] = []
    for idx, row in enumerate(top_rows, start=1):
        evidence_blocks.append(
            "\n".join(
                [
                    f"[{idx}] doc={row['doc_id']} page={row['page_start']} section={row.get('section') or '-'} task={row.get('task_id') or '-'} step={row.get('step_id') or '-'} score={float(row['score']):.3f}",
                    row["text"][:2000],
                ]
            )
        )

    human_prompt = "\n\n".join(
        [
            f"Soru: {query}",
            "Asagidaki AMM kanitlarini kullanarak kisa ve net cevap ver.",
            "Her iddia kanitta gecmiyorsa o iddiayi verme.",
            "Mutlaka kaynak satiri ekle: [doc_id p.page task step]",
            abstain_policy,
            "KANITLAR:",
            "\n\n".join(evidence_blocks),
        ]
    )

    try:
        llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
    except Exception as exc:
        raise RuntimeError(f"Gemini cagrisi basarisiz: {exc}") from exc

    answer_text = (response.content or "").strip() if isinstance(response.content, str) else str(response.content)
    if not answer_text:
        return build_answer(query, evidence_rows)

    citations, evidence = _build_citations_and_evidence(evidence_rows)
    return AnswerPacket(answer=answer_text, abstained=False, citations=citations, evidence=evidence)
