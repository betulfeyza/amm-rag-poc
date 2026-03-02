# AMM Agentic RAG — Proof of Concept

A local, offline retrieval-augmented system for querying **Aircraft Maintenance Manual (AMM)** PDFs.
Ask a natural-language question; the system finds and returns the exact AMM passage with full source citations — no LLM, no API key, no internet connection required at query time.

---

## How It Works

```
PDF files
  └─► parse pages (PyMuPDF / pdfplumber)
        └─► extract tables & figures
              └─► chunk text (~3 800 chars each)
                    └─► extract cross-references (xrefs)
                          └─► build FAISS index  (dense, all-MiniLM-L6-v2)
                          └─► build BM25  index  (sparse, BM25Okapi)

Query
  └─► hybrid retrieval  (55 % dense + 45 % sparse)
        └─► xref expansion
              └─► top-k chunks → extractive answer + citations
```

No generative model is involved. The answer is always a verbatim AMM excerpt.
If no relevant chunk is found the system **abstains** instead of hallucinating.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.10+ |
| faiss-cpu | ≥ 1.8.0 |
| sentence-transformers | ≥ 3.0.0 |
| langchain / langchain-community | ≥ 0.2.0 |
| rank-bm25 | ≥ 0.2.2 |
| PyMuPDF | ≥ 1.24.0 |
| pdfplumber | ≥ 0.11.0 |
| pydantic | ≥ 2.7.0 |
| numpy | < 2 (required for torch/onnxruntime compatibility) |

Install all at once:

```bash
pip install -r requirements.txt
pip install "numpy<2"
```

> **Note:** The embedding model (`all-MiniLM-L6-v2`, ~90 MB) is downloaded automatically from HuggingFace on first run.

---

## Quick Start

### 1 — Add your AMM PDFs

Place PDF files in `AMM_Sample/`.
The default PDFs expected are:

```
AMM_Sample/06___086.PDF
AMM_Sample/20___086.PDF
```

You can override this by passing `--pdfs` to the ingest command.

### 2 — Build the index

```bash
python -m src.cli ingest
```

This parses the PDFs, chunks the text, and builds FAISS + BM25 indexes under `data/`.
Run this **once** (or whenever PDFs change).

### 3 — Query

```bash
# Plain text answer
python -m src.cli ask --q "lubrication task steps"

# JSON output with full evidence
python -m src.cli ask --q "hydraulic tube fitting torque" --json --show-evidence

# Custom number of results
python -m src.cli ask --q "o-ring replacement" --topk 5
```

---

## Project Structure

```
├── src/
│   ├── cli.py                 # Entry point: ingest / ask commands
│   ├── ingest.py              # Orchestrates the full ingestion pipeline
│   ├── parse_pdf.py           # PDF → page records
│   ├── chunking.py            # Pages → text chunks
│   ├── extract_tables_figures.py
│   ├── extract_xrefs.py       # Cross-reference edge extraction
│   ├── indexing.py            # FAISS + BM25 index build & load
│   ├── retriever.py           # Hybrid retrieval + xref expansion
│   ├── agent.py               # Chunk list → AnswerPacket (extractive)
│   ├── schemas.py             # Dataclasses (PageRecord, ChunkRecord, …)
│   ├── config.py              # Paths and tuning constants
│   └── io_utils.py            # JSONL read/write helpers
├── prompts/
│   ├── system_answering.txt   # System prompt (for future LLM integration)
│   └── abstain_policy.txt     # Abstain response format
├── tests/
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_grounding.py      # Abstain logic unit test
│   └── test_retrieval.py      # Xref expansion unit test
├── AMM_Sample/                # ⚠ PDF files — not included in repo (proprietary)
├── data/                      # ⚠ Generated at ingest time — not included in repo
└── requirements.txt
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 3 tests pass without needing the PDF files (grounding and xref tests use synthetic data).
The CLI test also passes when the index exists.

---

## Output Format

```json
{
  "answer": "AMM bulgusu: LUBRICATION FITTINGS - SERVICING ...\nKaynak: [20___086 p.117 task=20-10-24-000-801 step=1]",
  "abstained": false,
  "citations": [
    { "doc_id": "20___086", "page": 117, "section": "20-10-24", "task_id": "20-10-24-000-801", "step_id": "1" }
  ],
  "evidence": [
    { "chunk_id": "ch_20___086_00063", "score": 0.955, "text": "...", "figure_ids": [], "table_ids": [] }
  ]
}
```

---

## Design Decisions

- **No LLM at query time** — keeps the system fully deterministic and auditable, which is critical for aviation maintenance documentation.
- **Hybrid retrieval** — dense embeddings handle semantic similarity; BM25 handles exact AMM task/step codes.
- **Abstain-first policy** — the system never fabricates an answer.
- **Extractive response** — the output is always a verbatim AMM excerpt, preserving regulatory language.
