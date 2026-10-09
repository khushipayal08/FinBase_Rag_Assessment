import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""Quick test: verify the PDF can be read."""
from pathlib import Path

from src.ingest import extract_pages

pdf_path = Path("data/raw/sample_1.pdf")
print(f"Looking for PDF at: {pdf_path.absolute()}")

if not pdf_path.exists():
    print("❌ ERROR: PDF file not found!")
    print("   Please make sure sample_1.pdf is in data/raw/")
    raise SystemExit(1)

print(f"✅ File found. Size: {pdf_path.stat().st_size:,} bytes")

pages = extract_pages(pdf_path)
print(f"✅ PDF opened. Total pages: {len(pages)}")

print()
print("--- First 300 characters of page 1 ---")
print(pages[1][:300])

print()
print("--- First 300 characters of page 2 ---")
print(pages[2][:300])