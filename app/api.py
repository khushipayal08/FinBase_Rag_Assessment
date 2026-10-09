"""FastAPI backend. Auto Swagger docs at /docs."""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_settings
from src.embed import get_collection
from src.generate import generate_answer
from src.guardrails import ABSTENTION, confidence_ok, groundedness_check
from src.query_rewrite import rewrite
from src.retrieve import retrieve

logging.basicConfig(
    level=get_settings().log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("api")

app = FastAPI(title="FinBase RAG Assistant", version="1.0.0")

_convos: dict[str, list[dict]] = defaultdict(list)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    conversation_id: str | None = None


class Source(BaseModel):
    chunk_id: str
    section_number: str | None = None
    section_title: str | None = None
    chunk_type: str | None = None
    source_page: int | None = None
    faq_ids: list[str] | None = None
    text: str
    retrieval_score: float
    rerank_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float
    latency_ms: int
    groundedness: dict


@app.get("/health")
def health():
    try:
        n = get_collection().count()
        return {"status": "ok", "indexed_chunks": n}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/sources")
def sources(limit: int = 100):
    col = get_collection()
    data = col.get(include=["metadatas"], limit=limit)
    return {"count": len(data["ids"]), "items": data["metadatas"]}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    conv_id = req.conversation_id or str(uuid.uuid4())
    history = _convos[conv_id]

    try:
        standalone = rewrite(req.question, history)
        log.info("conv=%s q=%r rewritten=%r", conv_id, req.question, standalone)

        chunks = retrieve(standalone, mode="hybrid_rerank")
        passes, conf = confidence_ok(chunks)

        if not passes:
            answer = ABSTENTION
            sources: list[Source] = []
            g = {"passed": True}
        else:
            answer = generate_answer(standalone, chunks)
            g = groundedness_check(answer, chunks)
            if not g["passed"]:
                log.warning("groundedness failed: %s", g)
                answer = ABSTENTION
                sources = []
            else:
                sources = [
                    Source(
                        chunk_id=c.chunk_id,
                        section_number=str(c.metadata.get("section_number") or ""),
                        section_title=str(c.metadata.get("section_title") or ""),
                        chunk_type=str(c.metadata.get("chunk_type") or ""),
                        source_page=int(c.metadata.get("source_page", 0)) or None,
                        faq_ids=_json_list(c.metadata.get("faq_ids")),
                        text=c.text,
                        retrieval_score=round(c.retrieval_score, 4),
                        rerank_score=round(c.rerank_score, 4),
                    )
                    for c in chunks
                ]

        _convos[conv_id].append({"role": "user", "content": req.question})
        _convos[conv_id].append({"role": "assistant", "content": answer})

        return AskResponse(
            answer=answer,
            sources=sources,
            confidence=round(conf, 4),
            latency_ms=int((time.perf_counter() - t0) * 1000),
            groundedness=g,
        )
    except Exception as e:
        log.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(e))


def _json_list(v):
    import json
    if not v:
        return None
    if isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except Exception:
        return None