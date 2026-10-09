"""Conversation-aware query rewriting for follow-up questions."""
from __future__ import annotations

import logging
from typing import Iterable

from google.genai import types

from src.generate import get_llm_client

log = logging.getLogger(__name__)

REWRITE_PROMPT = """Rewrite the user's follow-up question into a fully
self-contained question, using the prior conversation for context.

Rules:
- Preserve every constraint from previous turns (product, tenure, amount,
  customer type).
- Do NOT answer the question.
- Do NOT add facts not present in the history.
- If the question is already standalone, return it unchanged.
- Output ONLY the rewritten question, nothing else.

Conversation so far:
{history}

Follow-up question: {question}

Rewritten standalone question:"""

_FOLLOWUP_MARKERS = (
    "and ", "what about", "how about", "for senior", "uska", "uske",
    "aur ", "same for",
)

def _is_followup(q: str) -> bool:
    ql = q.lower().strip()
    return len(ql.split()) <= 6 and any(m in ql for m in _FOLLOWUP_MARKERS)

def rewrite(question: str, history: Iterable[dict]) -> str:
    """Rewrite follow-up into standalone. Skips LLM if question looks standalone."""
    history = list(history)
    if not history:
        return question
    if len(question.split()) >= 8 or not _is_followup(question):
        return question

    hist_txt = "\n".join(f"{h['role']}: {h['content']}" for h in history[-4:])
    full_prompt = REWRITE_PROMPT.format(history=hist_txt, question=question)

    try:
        client = get_llm_client()
        s = get_settings()

        response = client.models.generate_content(
            model=s.llm_model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=200,
            ),
        )
        rewritten = response.text.strip()
        log.info("rewrote %r -> %r", question, rewritten)
        return rewritten or question
    except Exception as e:
        log.warning("rewrite failed, using raw question: %s", e)
        return question