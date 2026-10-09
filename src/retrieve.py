"""Four retrieval modes: dense, BM25, hybrid (RRF), and hybrid + rerank."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from rank_bm25 import BM25Okapi

from src.config import get_settings
from src.embed import embed_query, get_collection

log = logging.getLogger(__name__)

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    retrieval_score: float = 0.0
    rerank_score: float = 0.0

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score else self.retrieval_score


_bm25: BM25Okapi | None = None
_bm25_corpus: list[dict] | None = None


def _tokenize(s: str) -> list[str]:
    return re.findall(r"[\w\u20b9][\w\u20b9,.\-]*", s.lower())


def _build_bm25() -> None:
    global _bm25, _bm25_corpus
    if _bm25 is not None:
        return
    col = get_collection()
    data = col.get(include=["documents", "metadatas"])
    corpus = [
        {"chunk_id": cid, "text": doc, "metadata": meta}
        for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]
    _bm25_corpus = corpus
    _bm25 = BM25Okapi([_tokenize(c["text"]) for c in corpus])
    log.info("BM25 index built over %d chunks", len(corpus))


def retrieve_dense(query: str, k: int) -> list[RetrievedChunk]:
    col = get_collection()
    res = col.query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    out: list[RetrievedChunk] = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append(RetrievedChunk(cid, doc, meta, retrieval_score=1.0 - dist))
    return out


def retrieve_bm25(query: str, k: int) -> list[RetrievedChunk]:
    _build_bm25()
    assert _bm25 and _bm25_corpus
    scores = _bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(_bm25_corpus, scores), key=lambda x: -x[1])[:k]
    return [
        RetrievedChunk(c["chunk_id"], c["text"], c["metadata"], retrieval_score=float(s))
        for c, s in ranked
    ]


def _rrf(rank_lists: list[list[RetrievedChunk]], k: int, rrf_k: int) -> list[RetrievedChunk]:
    fused: dict[str, float] = {}
    store: dict[str, RetrievedChunk] = {}
    for lst in rank_lists:
        for rank, ch in enumerate(lst, start=1):
            fused[ch.chunk_id] = fused.get(ch.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            store.setdefault(ch.chunk_id, ch)
    ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
    out: list[RetrievedChunk] = []
    for cid, score in ranked:
        c = store[cid]
        c.retrieval_score = score
        out.append(c)
    return out


def retrieve_hybrid(query: str, k: int, pool: int = 20) -> list[RetrievedChunk]:
    s = get_settings()
    dense = retrieve_dense(query, pool)
    bm25 = retrieve_bm25(query, pool)
    return _rrf([dense, bm25], k=k, rrf_k=s.rrf_k)


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        s = get_settings()
        log.info("loading reranker %s", s.reranker_model)
        _reranker = CrossEncoder(s.reranker_model, max_length=512)
    return _reranker


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    if not chunks:
        return []
    model = _get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs, show_progress_bar=False)
    for c, s in zip(chunks, scores):
        c.rerank_score = float(s)
    return sorted(chunks, key=lambda c: -c.rerank_score)[:top_k]


def retrieve(query: str, mode: RetrievalMode = "hybrid_rerank") -> list[RetrievedChunk]:
    s = get_settings()
    log.info("retrieve mode=%s query=%r", mode, query)
    if mode == "dense":
        return retrieve_dense(query, s.top_k)
    if mode == "bm25":
        return retrieve_bm25(query, s.top_k)
    if mode == "hybrid":
        return retrieve_hybrid(query, s.top_k)
    pool = retrieve_hybrid(query, s.top_k, pool=max(s.top_k * 2, 20))
    return rerank(query, pool, s.rerank_top_k)