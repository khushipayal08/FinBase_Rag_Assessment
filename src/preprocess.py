"""Deterministic text cleaning + structure parsing for the FinBase KB."""
from __future__ import annotations

import logging
import re
import uuid
from typing import Final, Literal

log = logging.getLogger(__name__)

ChunkType = Literal["policy", "table", "faq", "exclusion", "boilerplate", "meta"]

DOC_CODE = "FB-POL-FDW-2026-V3"
DOC_EFFECTIVE = "2026-10-01"
TARGET_TOKENS = 450
OVERLAP_TOKENS = 50
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------- symbol fixes

SYMBOL_MAP: Final[dict[str, str]] = {
    "\u25a0": "\u20b9",   # ■ -> ₹
    "\u00a5": "\u20b9",   # ¥ -> ₹
    "RBl":    "RBI",
    "WLt-":   "WLT-",
    "\u00b1": "",         # stray ±
    # PDF font maps the rupee symbol to "I" - fix the common patterns
    "I1,":    "\u20b91,",
    "I5,":    "\u20b95,",
    "I10,":   "\u20b910,",
    "I40,":   "\u20b940,",
    "I50,":   "\u20b950,",
    "I100,":  "\u20b9100,",
    "I500,":  "\u20b9500,",
    "I1,000": "\u20b91,000",
    "I5,00,000": "\u20b95,00,000",
    "I10,000": "\u20b910,000",
    "I50,00,000": "\u20b950,00,000",
}
LATEX_PATTERNS: Final[list[tuple[str, str]]] = [
    (r"\\boxed\{\s*\\begin\{array\}\{[^}]*\}([^}]*)\\end\{array\}\s*\}", r"\1"),
    (r"\\boxed\{([^}]*)\}", r"\1"),
    (r"\\\(|\\\)", ""),
    (r"\\\[|\\\]", ""),
    (r"\^\{?\*\*([^*]+)\*\*\}?", r"\1"),
    (r"\*\*", ""),
    (r"\\%", "%"),
    (r"\\", ""),
    (r"\{array\}", ""),
    (r"\$", ""),
]

GLUED_HEADING = re.compile(
    r"(?<=[a-z\)\]])(?=(?:Section\s+\d+|Guideline\s+\d+\.\d+|"
    r"Q\d{3}:|Standard\s+FB-|##\s))"
)

MERGED_BULLET = re.compile(r"(?<=[.:;])\s*-\s*(?=Guideline\s+\d)")


# ---------------------------------------------------------------- primitives

def fix_symbols(text: str) -> str:
    for bad, good in SYMBOL_MAP.items():
        text = text.replace(bad, good)
    return text


def strip_latex(text: str) -> str:
    for pat, repl in LATEX_PATTERNS:
        text = re.sub(pat, repl, text)
    return text


def split_glued_headings(text: str) -> str:
    return GLUED_HEADING.sub("\n", text)


def split_merged_bullets(text: str) -> str:
    return MERGED_BULLET.sub("\n- ", text)


def normalize_rupee_amounts(text: str) -> str:
    text = re.sub(r"(?i)\brs\.?\s*", "\u20b9", text)
    text = re.sub(r"\u20b9\s+", "\u20b9", text)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_faq_boilerplate(text: str) -> tuple[str, dict[str, str]]:
    """Remove the identical 4-line trailer every FAQ item carries."""
    meta: dict[str, str] = {}
    patterns = {
        "instrument_category":    r"Instrument Category:\s*(.+?)(?=\n|$)",
        "risk_o_meter":           r"Risk-?o-?meter Rating:\s*(.+?)(?=\n|$)",
        "tax_implication":        r"Tax implication:\s*(.+?)(?=\n|$)",
        "verification_reference": r"Verification reference:\s*(.+?)(?=\n|$)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            meta[key] = m.group(1).strip()
            text = text.replace(m.group(0), "")
    return normalize_whitespace(text), meta


# ---------------------------------------------------------------- entrypoint

def clean_document(raw: str) -> str:
    """Apply the full cleaning chain to a raw extracted document."""
    before = len(raw)
    text = fix_symbols(raw)
    text = strip_latex(text)
    text = split_glued_headings(text)
    text = split_merged_bullets(text)
    text = normalize_rupee_amounts(text)
    text = reformat_broken_tables(text)
    text = normalize_whitespace(text)
    log.info("clean_document: %d -> %d chars", before, len(text))
    return text


# ---------------------------------------------------------------- chunk helpers

def _approx_tokens(s: str) -> int:
    return len(s) // CHARS_PER_TOKEN


def _new_chunk(
    text: str,
    *,
    section_number: str,
    section_title: str,
    chunk_type: ChunkType,
    source_page: int,
    extra: dict | None = None,
) -> dict:
    chunk = {
        "chunk_id": f"{chunk_type[:3].upper()}-{uuid.uuid4().hex[:8]}",
        "text": text.strip(),
        "doc_code": DOC_CODE,
        "doc_effective": DOC_EFFECTIVE,
        "section_number": section_number,
        "section_title": section_title,
        "chunk_type": chunk_type,
        "source_page": source_page,
        "source_file": "sample_1.pdf",
        "token_estimate": _approx_tokens(text),
    }
    if extra:
        chunk.update(extra)
    return chunk


# ---------------------------------------------------------------- FAQ parsing

_FAQ_RE = re.compile(
    r"Q(\d{3}):\s*(?P<question>.+?)\s*\n(?P<answer>.+?)"
    r"(?=\n\s*Q\d{3}:|\Z)",
    flags=re.DOTALL,
)


def parse_faqs(section_text: str, page: int) -> list[dict]:
    items: list[dict] = []
    for m in _FAQ_RE.finditer(section_text):
        qid = f"Q{m.group(1)}"
        q = m.group("question").strip()
        raw_a = m.group("answer").strip()
        answer, meta = strip_faq_boilerplate(raw_a)
        items.append({
            "chunk_id": f"FAQ-{qid}",
            "faq_id": qid,
            "question": q,
            "text": f"Q: {q}\nA: {answer}",
            "doc_code": DOC_CODE,
            "section_number": "23",
            "section_title": "Fixed Deposits & Wealth FAQ Directory",
            "chunk_type": "faq",
            "source_page": page,
            "source_file": "sample_1.pdf",
            "token_estimate": _approx_tokens(f"Q: {q}\nA: {answer}"),
            "_faq_meta": meta,
        })
    log.info("parsed %d FAQ items", len(items))
    return items


# ---------------------------------------------------------------- table rows

def chunk_asset_table(rows: list[list[str]], page: int,
                      section_number: str) -> list[dict]:
    if not rows:
        return []
    header = [h.strip() for h in rows[0]]
    out: list[dict] = []
    for r in rows[1:]:
        cells = [c.strip() for c in r]
        if len(cells) != len(header):
            continue
        line = " | ".join(f"{h}: {c}" for h, c in zip(header, cells))
        out.append(_new_chunk(
            line,
            section_number=section_number,
            section_title="Wealth Product Comparison Matrix",
            chunk_type="table",
            source_page=page,
            extra={"table_id": "ASSET_MATRIX", "row_key": cells[0]},
        ))
    return out


# ---------------------------------------------------------------- exclusions

def chunk_exclusions(section_text: str, page: int) -> list[dict]:
    bullets = re.split(r"\n\s*-\s+", section_text)
    out: list[dict] = []
    for b in bullets[1:]:
        if len(b.strip()) < 20:
            continue
        out.append(_new_chunk(
            b,
            section_number="22",
            section_title="Explicit Product Exclusions & Unsupported Services Policy",
            chunk_type="exclusion",
            source_page=page,
            extra={"polarity": "negative"},
        ))
    return out


# ---------------------------------------------------------------- fallback

def recursive_split(text: str, target: int = TARGET_TOKENS,
                    overlap: int = OVERLAP_TOKENS) -> list[str]:
    if _approx_tokens(text) <= target:
        return [text]
    target_chars = target * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN
    for sep in ("\n\n", "\n", ". "):
        if sep in text:
            parts, buf = [], ""
            for piece in text.split(sep):
                if len(buf) + len(piece) + len(sep) <= target_chars:
                    buf += piece + sep
                else:
                    if buf:
                        parts.append(buf.strip())
                    buf = piece + sep
            if buf:
                parts.append(buf.strip())
            if all(_approx_tokens(p) <= target * 1.3 for p in parts):
                return parts
    return [text[i:i + target_chars]
            for i in range(0, len(text), target_chars - overlap_chars)]
def reformat_broken_tables(text: str) -> str:
    """Repair PDF-extracted tables where every cell is on its own line.

    Detects the FinBase FD rate table pattern (tenure bucket followed by
    rates/compounding on separate lines) and rewrites each row as a single
    '... | ... | ...' line so LLMs can associate cells with their row.
    """
    # Match: "<tenure phrase>\n<rate>\n<rate>\n<compounding>" repeated
    pattern = re.compile(
        r"(\d+\s+days?\s+to\s+\d+\s+days?"
        r"|\d+\s+year[s]?\s+to\s+(?:less than\s+)?\d+\s+year[s]?"
        r"|\d+\s+year[s]?\s+to\s+\d+\s+year[s]?)"
        r"\s*\n\s*"
        r"(\d+\.\d+%\s*p\.a\.)"
        r"\s*\n\s*"
        r"(\d+\.\d+%\s*p\.a\.)"
        r"\s*\n\s*"
        r"([A-Za-z ]+?)(?=\n|$)",
        re.IGNORECASE,
    )

    def _flatten(m: re.Match) -> str:
        tenure = m.group(1).strip()
        regular = m.group(2).strip()
        senior = m.group(3).strip()
        comp = m.group(4).strip()
        return f"{tenure} | Regular: {regular} | Senior Citizen: {senior} | Compounding: {comp}"

    return pattern.sub(_flatten, text)
