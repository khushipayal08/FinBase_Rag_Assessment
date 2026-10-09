"""Two deduplication passes for the FinBase KB.

Pass A: collapse the near-identical Sections 6-20 into one canonical copy.
Pass B: collapse the 100 FAQ items into ~10 unique questions.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")
_PAREN = re.compile(r"\(.*?\)")


def normalize_question(q: str) -> str:
    """Aggressive normalization for FAQ grouping."""
    q = _PAREN.sub(" ", q)
    q = _PUNCT.sub(" ", q)
    q = _WS.sub(" ", q)
    return q.strip().lower()


def boilerplate_signature(text: str) -> str:
    """Signature for detecting template-identical sections."""
    body = re.sub(r"\d+", "#", text)
    body = _WS.sub(" ", body)
    return body.strip()[:2000]


@dataclass
class DedupResult:
    canonical_chunks: list[dict]
    dropped_count: int
    merge_map: dict[str, list[str]] = field(default_factory=dict)


def collapse_boilerplate(
    sections: list[dict],
    min_group_size: int = 3,
) -> DedupResult:
    """Group sections by template signature; keep one per group."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in sections:
        groups[boilerplate_signature(s["text"])].append(s)

    kept: list[dict] = []
    dropped = 0
    merge_map: dict[str, list[str]] = {}

    for sig, members in groups.items():
        if len(members) >= min_group_size and _looks_like_boilerplate(members[0]):
            members.sort(key=lambda m: m["section_number"])
            canonical = dict(members[0])
            secs = [m["section_number"] for m in members]
            canonical["section_range"] = f"{min(secs)}-{max(secs)}"
            canonical["section_number"] = canonical["section_range"]
            canonical["source_pages"] = sorted({m["source_page"] for m in members})
            canonical["duplicate_of_count"] = len(members)
            canonical["is_canonical"] = True
            kept.append(canonical)
            merge_map[canonical["chunk_id"]] = [m["chunk_id"] for m in members[1:]]
            dropped += len(members) - 1
            log.info("boilerplate collapse: %d copies -> 1 (sections %s)",
                     len(members), canonical["section_range"])
        else:
            kept.extend(members)

    return DedupResult(kept, dropped, merge_map)


def _looks_like_boilerplate(chunk: dict) -> bool:
    """Guard against over-collapsing: require the template's fingerprint."""
    t = chunk["text"]
    return (
        "Wealth Portfolio Risk" in t
        and "algorithmically determined risk suitability" in t
        and "Systematic withdrawal plan" in t
    )


def collapse_faqs(faq_items: list[dict]) -> DedupResult:
    """Merge FAQ items whose normalized question text is identical."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in faq_items:
        groups[normalize_question(item["question"])].append(item)

    kept: list[dict] = []
    dropped = 0
    merge_map: dict[str, list[str]] = {}

    for norm_q, members in groups.items():
        members.sort(key=lambda m: m["faq_id"])
        canonical = dict(members[0])
        canonical["faq_ids"] = [m["faq_id"] for m in members]
        canonical["faq_id"] = members[0]["faq_id"]
        canonical["duplicate_of_count"] = len(members)
        canonical["is_canonical"] = True
        kept.append(canonical)
        merge_map[canonical["chunk_id"]] = [m["faq_id"] for m in members[1:]]
        dropped += len(members) - 1

    log.info("FAQ collapse: %d items -> %d unique", len(faq_items), len(kept))
    return DedupResult(kept, dropped, merge_map)