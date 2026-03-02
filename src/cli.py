from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import build_answer
from .retriever import retrieve


def main() -> None:
    parser = argparse.ArgumentParser("AMM Agentic RAG PoC")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Parse PDFs and build indexes")
    p_ing.add_argument("--pdfs", nargs="*", default=None)

    p_ask = sub.add_parser("ask", help="Ask AMM question")
    p_ask.add_argument("--q", required=True)
    p_ask.add_argument("--topk", type=int, default=8)
    p_ask.add_argument("--show-evidence", action="store_true")
    p_ask.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.cmd == "ingest":
        from .ingest import run_ingest

        pdfs = [Path(p) for p in args.pdfs] if args.pdfs else None
        result = run_ingest(pdfs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    rows = retrieve(args.q, topk=args.topk)
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
