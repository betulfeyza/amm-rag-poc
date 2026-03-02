from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "amm"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
FAISS_DIR = INDEX_DIR / "faiss"
BM25_DIR = INDEX_DIR / "bm25"
ASSETS_DIR = DATA_DIR / "assets"
FIGURES_DIR = ASSETS_DIR / "figures"

DEFAULT_PDFS = [
    BASE_DIR / "AMM_Sample" / "06___086.PDF",
    BASE_DIR / "AMM_Sample" / "20___086.PDF",
]

CHUNK_MAX_CHARS = 3800
DENSE_WEIGHT = 0.55
SPARSE_WEIGHT = 0.45
MIN_SCORE = 0.2


def ensure_dirs() -> None:
    for p in [RAW_DIR, PROCESSED_DIR, FAISS_DIR, BM25_DIR, FIGURES_DIR]:
        p.mkdir(parents=True, exist_ok=True)
