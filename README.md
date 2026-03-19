# AMM Agentic RAG — Proof of Concept

A local, offline retrieval-augmented system for querying **Aircraft Maintenance Manual (AMM)** PDFs.
Ask a natural-language question; the system finds relevant AMM passages and returns grounded answers with full citations.
Default query flow is Excel-grounded and supports extractive / OpenAI / Gemini answer modes.

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

If no relevant chunk is found, the system **abstains** instead of hallucinating.

---

## Requirements

| Dependency | Version |
|---|---|
| Python | 3.10+ |
| langchain-openai | ≥ 0.1.0 (optional, for OpenAI answer mode) |
| langchain-google-genai | ≥ 2.0.0 (optional, for Gemini answer mode) |
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

# OpenAI-backed grounded answer (optional)
OPENAI_API_KEY=your_key_here python -m src.cli ask --q "wing aspect ratio nedir" --llm openai

# JSON output with full evidence
python -m src.cli ask --q "hydraulic tube fitting torque" --json --show-evidence

# Custom number of results
python -m src.cli ask --q "o-ring replacement" --topk 5

# Ask directly from auto-chunked Excel (Otomatik_Parcalama.xlsx)
python -m src.cli ask --q "06-10-00-220-007 wing sweep nedir?" --source excel --json --show-evidence

# Same, but with OpenAI answer phrasing on top of Excel evidence
python -m src.cli ask --q "06-10-00-220-007 wing sweep nedir?" --source excel --llm openai --openai-model gpt-4o-mini --json --show-evidence

# Compare the same question across multiple Excel datasets
python -m src.cli measure --q "What is the wing sweep in SUBTASK 06-10-00-220-007?" --excel-paths Otomatik_Parcalama.xlsx Elle_Parcalama.xlsx Mistral_Parcalama.xlsx --llm gemini --gemini-model gemini-2.5-flash --topk 5 --min-score 0.25 --expect "25.03"
```

Excel source options:
- `--excel-path Otomatik_Parcalama.xlsx` (default)
- `--excel-sheet <sheet_name>` to query a specific sheet
- `--include-header-footer` to include header/footer rows in retrieval

### Optional: OpenAI answer mode

If you want the answer phrased by an LLM (still grounded to retrieved AMM evidence):

```bash
export OPENAI_API_KEY="your_openai_api_key"
python -m src.cli ask --q "06-10-00-220-007 wing sweep nedir" --llm openai

# choose a model
python -m src.cli ask --q "vertical stabilizer height" --llm openai --openai-model gpt-4o-mini
```

You can also store the key in a project-local `.env` file (recommended):

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
python -m src.cli ask --q "06-10-00-220-007 wing sweep nedir" --source excel --llm openai --openai-model gpt-4o-mini
```

Notes:
- Retrieval and citations remain local to your indexed AMM data.
- Without `--llm openai`, the system keeps the deterministic extractive behavior.

### Optional: Gemini answer mode

If OpenAI quota is unavailable, you can use Gemini with the same Excel-grounded flow:

```bash
export GEMINI_API_KEY="your_gemini_api_key"
python -m src.cli ask --q "What is the wing sweep in SUBTASK 06-10-00-220-007?" --source excel --excel-path Otomatik_Parcalama.xlsx --llm gemini --gemini-model gemini-2.0-flash --json --show-evidence
```

You can also set `GEMINI_API_KEY` in `.env`.

---

## Project Structure

```
├── src/
│   ├── cli.py                 # Entry point: ingest / ask commands
│   ├── excel_retriever.py      # Excel-based BM25 retrieval
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
├── scripts/
│   ├── auto_chunk.py          # Standalone auto-chunking & Excel export script
│   └── mistral_ocr_chunk.py   # Mistral OCR-based page extraction script
├── sample pages/              # Reference PNG screenshots of processed AMM pages
├── AMM_Sample/                # ⚠ PDF files — not included in repo (proprietary)
├── data/                      # ⚠ Generated at ingest time — not included in repo
└── requirements.txt
```

---

## Automatic Chunking Script (`scripts/auto_chunk.py`)

A standalone utility that processes selected AMM PDF pages, detects text block
structure automatically, and exports the results to a formatted Excel workbook
(`Otomatik_Parcalama.xlsx`).

### What it does

| Step | Tool | Output |
|------|------|--------|
| Text block extraction | PyMuPDF (`fitz`) | Per-block text, font size, Y-coordinate |
| Block classification | Heuristic rules | Başlık / Gövde / Tablo / Header-Footer |
| Special marker detection | Regex | ⚠ WARNING, 📝 NOTE, 🔧 TASK/SUBTASK |
| Table extraction | Camelot (lattice → stream) | Structured table data |
| Excel export | pandas + openpyxl | Color-coded, multi-sheet workbook |

### Processed pages

| PDF | Page (1-based) | Footer label |
|-----|----------------|--------------|
| `06___086.PDF` | 9 | Page 201 |
| `20___086.PDF` | 50 | Page 202 |
| `20___086.PDF` | 59 | Page 211 |

### Usage

```bash
# Place AMM PDFs in data/raw/amm/ then run:
python scripts/auto_chunk.py
```

Output file is written to the project root as `Otomatik_Parcalama.xlsx`
(excluded from version control via `.gitignore`).

### Color coding in Excel

| Color | Meaning |
|-------|---------|
| 🟡 Gold | WARNING block |
| 🔵 Light Blue | NOTE block |
| 🟢 Light Green | TASK / SUBTASK block |
| 🟠 Salmon | Section Heading |
| 🔷 Blue (header row) | Column header |

---

## Mistral OCR Chunking (`scripts/mistral_ocr_chunk.py`)

Alternative chunking flow using Mistral OCR for the same AMM pages.

### What it does

- Renders target PDF pages to image (`PyMuPDF`)
- Sends each page image to Mistral OCR API
- Splits OCR text into paragraphs
- Saves outputs separately:
  - `Mistral_Parcalama.xlsx`
  - `Mistral_OCR_Parcalama.xlsx` (legacy copy)
  - `data/processed/mistral_ocr_raw.json`

### Usage

```bash
# .env içinde MISTRAL_API_KEY tanımlı olmalı
python scripts/mistral_ocr_chunk.py
```

### Hardware requirement

- Local GPU gerekmez (OCR cloud API üzerinde çalışır).
- MacBook Pro ile rahatça çalışır; internet bağlantısı + API key + kota yeterlidir.

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

- **Evidence-first answering** — retrieval is deterministic; LLM mode is optional and grounded on retrieved evidence.
- **Hybrid retrieval** — dense embeddings handle semantic similarity; BM25 handles exact AMM task/step codes.
- **Abstain-first policy** — the system never fabricates an answer.
- **Extractive response** — the output is always a verbatim AMM excerpt, preserving regulatory language.
