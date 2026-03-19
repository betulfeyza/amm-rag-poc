from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import build_answer, build_answer_gemini, build_answer_openai
from .excel_retriever import retrieve_from_excel
from .retriever import retrieve


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    root_env = Path(__file__).resolve().parent.parent / ".env"
    if root_env.exists():
        load_dotenv(dotenv_path=root_env, override=False)


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser("AMM Agentic RAG PoC")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Parse PDFs and build indexes")
    p_ing.add_argument("--pdfs", nargs="*", default=None)

    p_ask = sub.add_parser("ask", help="Ask AMM question")
    p_ask.add_argument("--q", required=True)
    p_ask.add_argument("--topk", type=int, default=8)
    p_ask.add_argument("--source", choices=["index", "excel"], default="excel")
    p_ask.add_argument("--excel-path", default="Otomatik_Parcalama.xlsx")
    p_ask.add_argument("--excel-sheet", default=None)
    p_ask.add_argument("--include-header-footer", action="store_true")
    p_ask.add_argument("--min-score", type=float, default=0.2)
    p_ask.add_argument("--graph-hops", type=int, default=1)
    p_ask.add_argument("--llm", choices=["extractive", "openai", "gemini"], default="openai")
    p_ask.add_argument("--openai-model", default=None)
    p_ask.add_argument("--gemini-model", default=None)
    p_ask.add_argument("--temperature", type=float, default=0.0)
    p_ask.add_argument("--show-evidence", action="store_true")
    p_ask.add_argument("--json", action="store_true")

    p_measure = sub.add_parser("measure", help="Measure the same question across multiple Excel sources")
    p_measure.add_argument("--q", required=True)
    p_measure.add_argument(
        "--excel-paths",
        nargs="+",
        default=["Otomatik_Parcalama.xlsx", "Elle_Parcalama.xlsx", "Mistral_Parcalama.xlsx"],
    )
    p_measure.add_argument("--topk", type=int, default=5)
    p_measure.add_argument("--min-score", type=float, default=0.25)
    p_measure.add_argument("--graph-hops", type=int, default=1)
    p_measure.add_argument("--llm", choices=["extractive", "openai", "gemini"], default="gemini")
    p_measure.add_argument("--openai-model", default=None)
    p_measure.add_argument("--gemini-model", default=None)
    p_measure.add_argument("--temperature", type=float, default=0.0)
    p_measure.add_argument("--expect", default=None, help="Optional expected phrase to check in answer/evidence")
    p_measure.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.cmd == "ingest":
        from .ingest import run_ingest

        pdfs = [Path(p) for p in args.pdfs] if args.pdfs else None
        result = run_ingest(pdfs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.cmd == "measure":
        if args.llm == "openai" and not os.getenv("OPENAI_API_KEY"):
            raise SystemExit(
                "OpenAI modu hatasi: OPENAI_API_KEY bulunamadi. Proje kokune .env dosyasi olusturup OPENAI_API_KEY=... ekleyin."
            )
        if args.llm == "gemini":
            gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not gemini_key:
                raise SystemExit(
                    "Gemini modu hatasi: GEMINI_API_KEY (veya GOOGLE_API_KEY) bulunamadi. Proje kokune .env dosyasina ekleyin."
                )
            os.environ.setdefault("GOOGLE_API_KEY", gemini_key)

        results = []
        expect_lower = (args.expect or "").lower().strip()
        for excel_path in args.excel_paths:
            entry = {"excel_path": excel_path}
            try:
                rows = retrieve_from_excel(
                    args.q,
                    excel_path=Path(excel_path),
                    topk=args.topk,
                    sheet_name=None,
                    exclude_header_footer=True,
                    min_score=args.min_score,
                    graph_hops=args.graph_hops,
                )
            except (FileNotFoundError, ValueError) as exc:
                entry["error"] = f"Excel kaynak hatasi: {exc}"
                results.append(entry)
                continue

            try:
                if args.llm == "openai":
                    packet = build_answer_openai(
                        args.q,
                        rows,
                        model=args.openai_model or "gpt-4o-mini",
                        temperature=args.temperature,
                    )
                elif args.llm == "gemini":
                    packet = build_answer_gemini(
                        args.q,
                        rows,
                        model=args.gemini_model or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash",
                        temperature=args.temperature,
                    )
                else:
                    packet = build_answer(args.q, rows)
            except RuntimeError as exc:
                entry["error"] = f"{args.llm} modu hatasi: {exc}"
                results.append(entry)
                continue

            out = packet.to_dict()
            top_score = None
            top_chunk = None
            if out.get("evidence"):
                top_score = out["evidence"][0].get("score")
                top_chunk = out["evidence"][0].get("chunk_id")

            entry.update(
                {
                    "abstained": out.get("abstained"),
                    "answer": out.get("answer"),
                    "citations": out.get("citations", []),
                    "evidence_count": len(out.get("evidence", [])),
                    "top_score": top_score,
                    "top_chunk": top_chunk,
                }
            )

            if expect_lower:
                hay = (out.get("answer") or "") + "\n" + "\n".join(
                    [str(e.get("text", "")) for e in out.get("evidence", [])]
                )
                entry["expect_match"] = expect_lower in hay.lower()

            results.append(entry)

        payload = {
            "query": args.q,
            "llm": args.llm,
            "topk": args.topk,
            "min_score": args.min_score,
            "results": results,
        }

        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"query={payload['query']}")
            print(f"llm={payload['llm']} topk={payload['topk']} min_score={payload['min_score']}")
            for item in results:
                print(f"\n=== {item['excel_path']} ===")
                if item.get("error"):
                    print(f"error={item['error']}")
                    continue
                print(f"abstained={item['abstained']} evidence_count={item['evidence_count']} top_score={item['top_score']}")
                if "expect_match" in item:
                    print(f"expect_match={item['expect_match']}")
                print(item["answer"])
        return

    if args.source == "excel":
        try:
            rows = retrieve_from_excel(
                args.q,
                excel_path=Path(args.excel_path),
                topk=args.topk,
                sheet_name=args.excel_sheet,
                exclude_header_footer=not args.include_header_footer,
                min_score=args.min_score,
                graph_hops=args.graph_hops,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"Excel kaynak hatasi: {exc}")
    else:
        rows = retrieve(args.q, topk=args.topk)
    if args.llm in {"openai", "gemini"} and args.source != "excel":
        raise SystemExit("OpenAI/Gemini modu yalnizca Excel kaynagi ile calisir. --source excel kullanin.")

    if args.llm == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OpenAI modu hatasi: OPENAI_API_KEY bulunamadi. Proje kokune .env dosyasi olusturup OPENAI_API_KEY=... ekleyin.")
        try:
            packet = build_answer_openai(
                args.q,
                rows,
                model=args.openai_model or "gpt-4o-mini",
                temperature=args.temperature,
            )
        except RuntimeError as exc:
            raise SystemExit(f"OpenAI modu hatasi: {exc}")
    elif args.llm == "gemini":
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            raise SystemExit("Gemini modu hatasi: GEMINI_API_KEY (veya GOOGLE_API_KEY) bulunamadi. Proje kokune .env dosyasina ekleyin.")
        os.environ.setdefault("GOOGLE_API_KEY", gemini_key)
        try:
            packet = build_answer_gemini(
                args.q,
                rows,
                model=args.gemini_model or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash",
                temperature=args.temperature,
            )
        except RuntimeError as exc:
            raise SystemExit(f"Gemini modu hatasi: {exc}")
    else:
        packet = build_answer(args.q, rows)
    out = packet.to_dict()

    if not args.show_evidence:
        out["evidence"] = []

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(out["answer"])
        print(f"abstained={out['abstained']}")
        print("citations:")
        for c in out["citations"]:
            print(f"- {c['doc_id']} p.{c['page']} task={c.get('task_id')} step={c.get('step_id')}")
        if args.show_evidence:
            print("evidence:")
            for e in out["evidence"]:
                print(f"- {e['chunk_id']} score={e['score']:.3f}")


if __name__ == "__main__":
    main()
