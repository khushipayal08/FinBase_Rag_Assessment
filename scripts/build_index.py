"""End-to-end index build: PDF -> chunks -> Chroma."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

from src.chunk import build_all_chunks
from src.embed import index_chunks
from src.ingest import extract_pages
from src.preprocess import clean_document

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)s | %(message)s")


def main() -> None:
    raw_pdf = Path("data/raw/sample_1.pdf")
    if not raw_pdf.exists():
        print(f"ERROR: {raw_pdf} not found. Put sample_1.pdf in data/raw/")
        raise SystemExit(1)

    print("Step 1/4: Extracting pages from PDF...")
    pages = extract_pages(raw_pdf)
    print(f"   -> {len(pages)} pages extracted")

    print("Step 2/4: Cleaning text...")
    cleaned = {p: clean_document(t) for p, t in pages.items()}

    print("Step 3/4: Building chunks (dedup + structure-aware split)...")
    chunks = build_all_chunks(cleaned)

    original_count = len(chunks)
    chunks = [c for c in chunks if c.get("text") and c["text"].strip()]
    if len(chunks) != original_count:
        print(f"   -> Filtered {original_count - len(chunks)} empty chunks")

    print(f"   -> {len(chunks)} chunks after dedup")

    print("Step 4/4: Embedding + indexing into Chroma...")
    n = index_chunks(chunks)
    print(f"\nDONE. Indexed {n} chunks.")
    print(f"Chroma DB saved at: {Path('data/chroma').absolute()}")


if __name__ == "__main__":
    main()