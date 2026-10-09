"""Dense embeddings via BGE-small-en-v1.5, plus Chroma persistence."""
from __future__ import annotations

import logging

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from src.config import get_settings

log = logging.getLogger(__name__)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        s = get_settings()
        log.info("loading embedding model %s", s.embedding_model)
        _model = SentenceTransformer(s.embedding_model)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    # Filter out empty/whitespace-only texts — empty lists crash the encoder.
    clean_texts = [t for t in texts if t and t.strip() and len(t.strip()) > 3]
    log.info("embed_documents: received %d, kept %d", len(texts), len(clean_texts))

    if not clean_texts:
        log.error("No non-empty texts to embed!")
        return []

    model = _get_model()
    try:
        result = model.encode(
            clean_texts,
            normalize_embeddings=True,
            batch_size=4,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return result.tolist()
    except Exception as e:
        log.exception("encode failed for %d texts: %s", len(clean_texts), e)
        raise


def embed_query(text: str) -> list[float]:
    return _get_model().encode(
        [BGE_QUERY_PREFIX + text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()


def get_client() -> chromadb.ClientAPI:
    s = get_settings()
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(s.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection(name: str = "finbase") -> chromadb.Collection:
    return get_client().get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"}
    )


def index_chunks(chunks: list[dict], collection_name: str = "finbase") -> int:
    """Wipe + rebuild. Idempotent for reproducible evaluation."""
    # Pre-filter: drop chunks with empty text
    chunks = [c for c in chunks if c.get("text") and c["text"].strip()]
    log.info("index_chunks: %d chunks after empty-filter", len(chunks))

    if not chunks:
        raise ValueError("No chunks to index!")

    client = get_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metadatas = [_sanitize_meta(c) for c in chunks]

    vectors = embed_documents(docs)

    if len(vectors) != len(ids):
        raise RuntimeError(
            f"Embedding count mismatch: {len(vectors)} vectors for {len(ids)} chunks"
        )

    col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=vectors)
    log.info("indexed %d chunks into '%s'", len(chunks), collection_name)
    return len(chunks)


def _sanitize_meta(c: dict) -> dict:
    """Chroma only accepts str/int/float/bool — JSON-encode the rest."""
    import json
    out: dict = {}
    for k, v in c.items():
        if k == "text":
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out
def ensure_index() -> int:
    """Build the Chroma index if it's empty. Returns chunk count.

    Streamlit Cloud deployment doesn't ship data/chroma/ (gitignored),
    so we rebuild on first run. Subsequent runs reuse the cache.
    """
    import logging
    log = logging.getLogger(__name__)

    try:
        col = get_collection()
        count = col.count()
        if count > 0:
            log.info("Chroma index ready: %d chunks", count)
            return count
    except Exception as e:
        log.warning("Could not read Chroma count: %s", e)

    log.info("Chroma index empty — building from PDF...")

    # Import here to avoid circular imports
    from pathlib import Path
    from src.ingest import extract_pages
    from src.preprocess import clean_document
    from src.chunk import build_all_chunks

    pdf = Path("data/raw/sample_1.pdf")
    if not pdf.exists():
        log.error("PDF not found at %s", pdf)
        return 0

    pages = extract_pages(pdf)
    cleaned = {p: clean_document(t) for p, t in pages.items()}
    chunks = build_all_chunks(cleaned)
    chunks = [c for c in chunks if c.get("text") and c["text"].strip()]

    return index_chunks(chunks)