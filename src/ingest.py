"""Multi-strategy PDF extraction with OCR fallback."""
from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

log = logging.getLogger(__name__)


def extract_pages(pdf: Path) -> dict[int, str]:
    """Extract per-page text; fall back to OCR if a page looks degenerate.

    Degenerate = >40% of tokens are single characters (the '1 1 1 1' case).
    """
    pages: dict[int, str] = {}
    with fitz.open(pdf) as doc:
        for i, page in enumerate(doc, start=1):
            txt = page.get_text("text")
            if _is_degenerate(txt):
                log.warning("page %d degenerate, attempting OCR", i)
                txt = _ocr_page(page) or txt
            pages[i] = txt
    return pages


def _is_degenerate(txt: str) -> bool:
    toks = txt.split()
    if len(toks) < 20:
        return False
    singles = sum(1 for t in toks if len(t) == 1 and t.isdigit())
    return singles / len(toks) > 0.4


def _ocr_page(page: "fitz.Page", dpi: int = 300) -> str | None:
    try:
        import io
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception as e:
        log.warning("OCR failed: %s", e)
        return None