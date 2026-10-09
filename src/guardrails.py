"""Hallucination mitigation: confidence gate + post-generation groundedness."""
from __future__ import annotations

import logging
import re

from src.config import get_settings
from src.retrieve import RetrievedChunk

log = logging.getLogger(__name__)

ABSTENTION = "I couldn't find this information in the FinBase knowledge base."

_NUMBER_RE = re.compile(r"₹?\s?\d[\d,]*(?:\.\d+)?%?")
_CITATION_RE = re.compile(
    r"Section\s+(\d+(?:-\d+)?)(?:,\s*FAQ\s+(Q\d{3}))?", re.IGNORECASE
)


def confidence_ok(chunks: list[RetrievedChunk]) -> tuple[bool, float]:
    """Return (passes_gate, top_score)."""
    if not chunks:
        return False, 0.0
    top = max(c.final_score for c in chunks)
    passes = top >= get_settings().confidence_threshold
    log.info("confidence gate: top=%.3f threshold=%.3f passes=%s",
             top, get_settings().confidence_threshold, passes)
    return passes, top


def numbers_in(text: str) -> set[str]:
    """Extract comparable numeric tokens from text."""
    out: set[str] = set()
    for m in _NUMBER_RE.findall(text):
        normalized = m.replace(" ", "").replace("₹", "")
        out.add(normalized.replace(",", ""))
    return out


def groundedness_check(answer: str, chunks: list[RetrievedChunk]) -> dict:
    """Return a structured report. Never raises.

    Important: section numbers like "Section 23" or FAQ IDs like "Q001"
    appear in LLM answers as *citations*, not as facts. They live in chunk
    metadata, not in the chunk text. We add them to the allowed-numbers
    set so the grounding check doesn't false-reject correct answers.
    """
    context_text = "\n".join(c.text for c in chunks)
    ctx_nums = numbers_in(context_text)

    # Allow numbers that appear in metadata (section numbers, FAQ IDs)
    for c in chunks:
        sec = str(c.metadata.get("section_number", ""))
        # Section "6-20" -> add both 6 and 20
        for tok in re.findall(r"\d+", sec):
            ctx_nums.add(tok)
        # FAQ IDs like "Q001,Q011" -> add 001, 011, and 1, 11
        faq_ids = str(c.metadata.get("faq_ids", ""))
        for qid in re.findall(r"Q(\d+)", faq_ids):
            ctx_nums.add(qid)          # "001"
            ctx_nums.add(str(int(qid)))  # "1"
        # Source page numbers are fine too
        page = str(c.metadata.get("source_page", ""))
        if page:
            ctx_nums.add(page)

    ans_nums = numbers_in(answer)
    unsupported = sorted(n for n in ans_nums if n not in ctx_nums)

    retrieved_sections = {str(c.metadata.get("section_number", "")) for c in chunks}
    cited = _CITATION_RE.findall(answer)
    bad_citations = []
    for sec, _faq in cited:
        if sec in retrieved_sections:
            continue
        if "-" in sec:
            lo, hi = map(int, sec.split("-"))
            if any(s.isdigit() and lo <= int(s) <= hi for s in retrieved_sections):
                continue
        bad_citations.append(sec)

    passed = not unsupported and not bad_citations
    report = {
        "passed": passed,
        "unsupported_numbers": unsupported,
        "bad_citations": bad_citations,
        "n_numbers_in_answer": len(ans_nums),
        "n_numbers_in_context": len(ctx_nums),
    }
    log.info("groundedness: %s", report)
    return report

    retrieved_sections = {str(c.metadata.get("section_number", "")) for c in chunks}
    cited = _CITATION_RE.findall(answer)
    bad_citations = []
    for sec, _faq in cited:
        if sec in retrieved_sections:
            continue
        if "-" in sec:
            lo, hi = map(int, sec.split("-"))
            if any(s.isdigit() and lo <= int(s) <= hi for s in retrieved_sections):
                continue
        bad_citations.append(sec)

    passed = not unsupported and not bad_citations
    report = {
        "passed": passed,
        "unsupported_numbers": unsupported,
        "bad_citations": bad_citations,
        "n_numbers_in_answer": len(ans_nums),
        "n_numbers_in_context": len(ctx_nums),
    }
    log.info("groundedness: %s", report)
    return report