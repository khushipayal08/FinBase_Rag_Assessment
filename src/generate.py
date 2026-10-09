"""Grounded answer generation with Gemini 2.0 Flash (new google-genai SDK)."""
from __future__ import annotations

import logging
from typing import Any, Iterable

from google import genai
from google.genai import types

from src.config import get_settings
from src.retrieve import RetrievedChunk

log = logging.getLogger(__name__)

_client = None

def get_llm_client():
    """Lazy-init the Gemini client with the new SDK."""
    global _client
    if _client is None:
        s = get_settings()
        if not s.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")
        _client = genai.Client(api_key=s.gemini_api_key)
        log.info("Gemini client initialized (google-genai SDK).")
    return _client

SYSTEM_PROMPT = """You are a FinBase customer support assistant.

STRICT RULES:
1. Answer ONLY using the CONTEXT below. Never use outside knowledge.
2. If the CONTEXT does not contain the answer, reply EXACTLY:
   "I couldn't find this information in the FinBase knowledge base."
3. NEVER invent, round, or approximate rates, fees, limits, or numbers.
4. Quote numbers exactly as they appear in the context.
5. If the CONTEXT contains conflicting values for the same fact, report BOTH
   values and explicitly say the knowledge base has conflicting information.
6. If the user asks about an unsupported service (Section 22), state clearly
   that FinBase does NOT offer it and cite Section 22.
7. If the user asks about a section not present in the context (e.g. Section
   21), abstain with the exact sentence from rule 2.
8. If the user asserts a fact contradicting the context, correct them using
   the context and cite the source. Never agree with a wrong premise.
9. Always end with source citations in this exact format:

Answer: <your answer>

Source: <Document name> — Section <number>
   For FAQ: Source: <Document name> — Section 23, FAQ <QID>
   For multiple: list each on its own line prefixed with "- "

CONTEXT:
{context}
"""

def _format_context(chunks: Iterable[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        sec = c.metadata.get("section_number", "?")
        title = c.metadata.get("section_title", "")
        ctype = c.metadata.get("chunk_type", "")
        faq_ids = c.metadata.get("faq_ids", "")
        header = f"[{i}] Section {sec} ({title}) [{ctype}]"
        if faq_ids:
            header += f" FAQ IDs: {faq_ids}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)

def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Generate a grounded answer using only the retrieved chunks."""
    if not chunks:
        return "I couldn't find this information in the FinBase knowledge base."

    context = _format_context(chunks)
    full_prompt = SYSTEM_PROMPT.format(context=context) + f"\n\nUser: {question}\n"

    client = get_llm_client()
    s = get_settings()

    # Build config. thinking_config is optional — attach it only if the
    # installed SDK exposes ThinkingConfig (varies by google-genai version).
    config_kwargs: dict[str, Any] = {
        "temperature": 0,
        "top_p": 1,
        "max_output_tokens": 2048,
    }
    thinking_cfg_cls = getattr(types, "ThinkingConfig", None)
    if thinking_cfg_cls is not None:
        try:
            config_kwargs["thinking_config"] = thinking_cfg_cls(thinking_budget=0)
        except Exception:
            pass

    config = types.GenerateContentConfig(**config_kwargs)

        # Retry with exponential backoff for transient 503/429 errors.
    import time
    max_attempts = 4
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=s.llm_model,
                contents=full_prompt,
                config=config,
            )
            return response.text
        except Exception as e:
            last_error = e
            err_str = str(e)
            # Retry only on transient errors (503 busy, 429 rate limit)
            transient = (
                "503" in err_str
                or "UNAVAILABLE" in err_str
                or "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
            )
            if not transient or attempt == max_attempts:
                log.error("Gemini failed after %d attempts: %s", attempt, e)
                break
            wait = 2 ** (attempt - 1)   # 1s, 2s, 4s
            log.warning(
                "Gemini attempt %d failed (transient), retrying in %ds: %s",
                attempt, wait, err_str[:120],
            )
            time.sleep(wait)

    # All retries exhausted — return a friendly message, not a crash.
    log.exception("Gemini generation failed permanently: %s", last_error)
    return (
        "I'm temporarily unable to reach the FinBase AI service "
        "(the language model is experiencing high demand). "
        "Please try your question again in a few moments."
    )