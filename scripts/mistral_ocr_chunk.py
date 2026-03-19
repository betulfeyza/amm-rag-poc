"""
Mistral OCR ile AMM sayfalarını parçalama scripti
-------------------------------------------------
Hedef:
  - 06___086.PDF -> Sayfalar 9, 10, 11 (0-based: 8, 9, 10)
    - 20___086.PDF -> Sayfalar 50, 59 (0-based: 49, 58)

Çıktılar:
    - Mistral_Parcalama.xlsx
    - Mistral_OCR_Parcalama.xlsx (legacy)
  - data/processed/mistral_ocr_raw.json

Gereksinimler:
  - MISTRAL_API_KEY ortam değişkeni (.env içinde olabilir)
  - internet bağlantısı
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pandas as pd
import requests
from dotenv import dotenv_values, load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_XLSX = BASE_DIR / "Mistral_Parcalama.xlsx"
OUTPUT_XLSX_LEGACY = BASE_DIR / "Mistral_OCR_Parcalama.xlsx"
OUTPUT_RAW_JSON = BASE_DIR / "data" / "processed" / "mistral_ocr_raw.json"

MISTRAL_API_URL = "https://api.mistral.ai/v1/ocr"
MISTRAL_MODEL = "mistral-ocr-latest"

TARGETS: dict[str, list[int]] = {
    "06___086.PDF": [8, 9, 10],  # 9,10,11
    "20___086.PDF": [49, 58],    # 50,59 -> footer 202,211
}


def resolve_pdf_dir(base_dir: Path) -> Path:
    candidates = [
        base_dir / "data" / "raw" / "amm",
        base_dir / "files",
        base_dir / "AMM_Sample",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def detect_footer_page(text: str) -> str | None:
    m = re.search(r"\bPage\s+(\d{1,4})\b", text or "", re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text or "")]
    return [p for p in parts if p]


def render_page_to_data_uri(pdf_path: Path, page_idx: int, zoom: float = 2.0) -> str:
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_idx]
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pix.tobytes("png")
    finally:
        doc.close()

    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def call_mistral_ocr(image_data_uri: str, api_key: str, model: str) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "document": {
            "type": "image_url",
            "image_url": image_data_uri,
        },
    }

    response = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def extract_text_from_response(ocr_json: dict[str, Any]) -> str:
    pages = ocr_json.get("pages") or []
    if pages and isinstance(pages, list):
        first = pages[0] or {}
        text = first.get("markdown") or first.get("text") or ""
        return str(text)

    output_text = ocr_json.get("output_text")
    if output_text:
        return str(output_text)

    return ""


def main() -> None:
    env_path = BASE_DIR / ".env"
    load_dotenv(env_path, override=False)

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key and env_path.exists():
        api_key = str(dotenv_values(env_path).get("MISTRAL_API_KEY") or "").strip()

    if not api_key:
        raise SystemExit("MISTRAL_API_KEY bulunamadi. .env dosyasina ekleyin.")

    pdf_dir = resolve_pdf_dir(BASE_DIR)

    print("=" * 60)
    print("Mistral OCR parçalama başlıyor...")
    print("=" * 60)
    print(f"PDF klasörü: {pdf_dir}")

    rows: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []
    row_id = 1

    for pdf_name, page_indices in TARGETS.items():
        pdf_path = pdf_dir / pdf_name
        if not pdf_path.exists():
            print(f"⚠ PDF bulunamadi: {pdf_path}")
            continue

        for page_idx in page_indices:
            print(f"\n[{pdf_name}] Sayfa {page_idx + 1} OCR işleniyor...")
            try:
                image_data_uri = render_page_to_data_uri(pdf_path, page_idx)
                ocr_json = call_mistral_ocr(image_data_uri, api_key=api_key, model=MISTRAL_MODEL)
                text = extract_text_from_response(ocr_json)
            except Exception as exc:
                print(f"  ⚠ OCR hatasi (sayfa {page_idx + 1}): {exc}")
                continue

            footer_page = detect_footer_page(text)
            paras = split_paragraphs(text)
            print(f"  Paragraf sayisi: {len(paras)}")

            raw_pages.append(
                {
                    "pdf": pdf_name,
                    "pdf_page": page_idx + 1,
                    "footer_page": footer_page,
                    "text": text,
                    "response": ocr_json,
                }
            )

            for p in paras:
                rows.append(
                    {
                        "ID": f"MISTRAL_{row_id:04d}",
                        "İçerik": p,
                        "Kaynak PDF": pdf_name,
                        "Sayfa (PDF)": page_idx + 1,
                        "Sayfa (Footer)": footer_page or "?",
                        "Model": MISTRAL_MODEL,
                        "Kaynak": "Mistral OCR",
                    }
                )
                row_id += 1

    if not rows:
        print("\n❌ İşlenecek OCR çıktısı oluşmadı.")
        return

    df = pd.DataFrame(rows)
    df.to_excel(OUTPUT_XLSX, index=False)
    df.to_excel(OUTPUT_XLSX_LEGACY, index=False)

    OUTPUT_RAW_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RAW_JSON.write_text(json.dumps(raw_pages, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Tamamlandı: {OUTPUT_XLSX}")
    print(f"✅ Legacy çıktı: {OUTPUT_XLSX_LEGACY}")
    print(f"✅ Ham OCR JSON: {OUTPUT_RAW_JSON}")


if __name__ == "__main__":
    main()
