"""Test retrieval: verify chunks are searchable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieve import retrieve

test_queries = [
    "What is the interest rate for senior citizens?",
    "What is the penalty for breaking an FD early?",
    "What is the minimum SIP amount?",
    "Does FinBase offer cryptocurrency?",
    "TDS threshold for senior citizens",
]

for q in test_queries:
    print(f"\n{'='*70}")
    print(f"QUERY: {q}")
    print('='*70)
    chunks = retrieve(q, mode="hybrid_rerank")
    for i, c in enumerate(chunks[:3], 1):
        sec = c.metadata.get("section_number", "?")
        title = c.metadata.get("section_title", "")
        ctype = c.metadata.get("chunk_type", "")
        preview = c.text[:120].replace("\n", " ")
        print(f"\n  [{i}] Section {sec} ({ctype}) | rerank: {c.rerank_score:.3f}")
        print(f"      {title}")
        print(f"      {preview}...")