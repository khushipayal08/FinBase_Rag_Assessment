"""Entry point that ties parsing -> dedup -> chunking together."""
from __future__ import annotations

import logging
import re

from src.preprocess_dedup import collapse_boilerplate, collapse_faqs
from src.preprocess import (
    chunk_asset_table,
    chunk_exclusions,
    parse_faqs,
    _new_chunk,
    recursive_split,
    _approx_tokens,
)

log = logging.getLogger(__name__)


# Flexible section heading matcher. Handles:
#   ## Section 2: Title
#   Section 2: Title
#   Section 2 - Title
#   Section 2. Title
#   2. Title  (numbered fallback, only if "Section" keyword absent)
_SECTION_HEADING_RE = re.compile(
    r"(?:^|\n)[ \t]*"
    r"(?:#+[ \t]*)?"
    r"Section[ \t]+(\d+)"
    r"[ \t]*[:\-–—.]?[ \t]*"
    r"([^\n]*)",
    re.MULTILINE | re.IGNORECASE,
)


def _split_sections(full_text: str) -> list[dict]:
    """Find all section headings and split text between them."""
    matches = list(_SECTION_HEADING_RE.finditer(full_text))
    if not matches:
        log.warning("No 'Section N' headings found in text")
        return []

    sections: list[dict] = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        title_from_heading = m.group(2).strip().strip("#: -–—.")
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        sections.append({
            "section_number": str(num),
            "section_title": title_from_heading or f"Section {num}",
            "text": body,
        })

    log.info("found %d section headings: %s",
             len(sections), [s["section_number"] for s in sections])
    return sections


def build_all_chunks(pages: dict[int, str]) -> list[dict]:
    full = "\n\n".join(pages[p] for p in sorted(pages))

    raw_sections = _split_sections(full)
    if not raw_sections:
        log.error("Cannot proceed: 0 sections found.")
        return []

    # Attach page number + chunk_type + chunk_id
    sections: list[dict] = []
    for s in raw_sections:
        num = int(s["section_number"])
        sections.append({
            "chunk_id": f"SEC-{num:02d}",
            "section_number": s["section_number"],
            "section_title": s["section_title"],
            "text": s["text"],
            "source_page": _guess_page(num, pages),
            "chunk_type": _classify(num, s["text"]),
        })

    # Deduplicate: boilerplate sections
    r1 = collapse_boilerplate(sections)

    # Deduplicate: FAQ items
    faq_raw: list[dict] = []
    non_faq: list[dict] = []
    for s in r1.canonical_chunks:
        if s["section_number"] == "23":
            faq_raw.extend(parse_faqs(s["text"], s["source_page"]))
        else:
            non_faq.append(s)
    r2 = collapse_faqs(faq_raw)

    # Build final list
    out: list[dict] = []
    for s in non_faq:
        if s["chunk_type"] == "exclusion":
            out.extend(chunk_exclusions(s["text"], s["source_page"]))
        elif _approx_tokens(s["text"]) > 600:
            for i, sub in enumerate(recursive_split(s["text"])):
                out.append(_new_chunk(
                    sub,
                    section_number=s["section_number"],
                    section_title=s["section_title"],
                    chunk_type=s["chunk_type"],
                    source_page=s["source_page"],
                    extra={"sub_chunk_index": i},
                ))
        else:
            out.append(s)
    out.extend(r2.canonical_chunks)

    log.info(
        "final chunks: %d (boilerplate dropped: %d, FAQ dropped: %d)",
        len(out), r1.dropped_count, r2.dropped_count,
    )
    return out


def _classify(num: int, body: str) -> str:
    if num == 22:
        return "exclusion"
    if num == 23:
        return "faq"
    return "policy"


def _guess_page(num: int, pages: dict[int, str]) -> int:
    needle_patterns = [f"Section {num}", f"Section {num}:", f"Section {num} "]
    for p, t in pages.items():
        for needle in needle_patterns:
            if needle in t:
                return p
    return 1