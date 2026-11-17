"""LLM-as-judge for answer faithfulness (RAGAS 'faithfulness' equivalent).

Uses the same Groq model as generation (gpt-oss-120b) with temperature 0.
The judge answers two YES/NO questions about each generated answer:
1. faithful — is every claim in the answer supported by the retrieved context?
2. answers_question — does the answer actually answer the question?

Returns {"faithful": bool|None, "answers_question": bool|None,
         "reason": str, "error": str|None}.
A judge failure (API error / unparsable output) returns None values with
an error string — it never crashes the eval run.
"""

import json
import logging

import logfire
from groq import Groq

from app.config import settings

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of a retrieval-augmented QA system.
You judge one generated answer against the context it was built from.
Reply with ONLY a JSON object, no other text, no markdown fences."""

JUDGE_USER_TEMPLATE = """QUESTION:
{question}

CONTEXT (retrieved document chunks):
{context}

ANSWER:
{answer}

Decide two things:
1. faithful: is EVERY claim in the answer supported by the context above? Answer YES only if nothing in the answer goes beyond the context.
2. answers_question: does the answer actually answer the question asked?

Reply with ONLY this JSON object:
{{"faithful": "YES", "answers_question": "YES", "reason": "<one sentence>"}}"""


def _get_clients():
    """Primary + fallback Groq clients (fallback skipped when its key is empty)."""
    clients = []
    if settings.GROQ_API_KEY:
        clients.append(Groq(api_key=settings.GROQ_API_KEY))
    if getattr(settings, "GROQ_FALLBACK_API_KEY", ""):
        clients.append(Groq(api_key=settings.GROQ_FALLBACK_API_KEY))
    return clients


def _parse_verdict(raw: str):
    """Parse the judge's JSON output; tolerates markdown code fences."""
    text = raw.strip()
    if not text:
        raise ValueError("empty judge response")
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    data = json.loads(text)
    return {
        "faithful": str(data["faithful"]).strip().upper() == "YES",
        "answers_question": str(data["answers_question"]).strip().upper() == "YES",
        "reason": str(data.get("reason", "")),
        "error": None,
    }


@logfire.instrument("judge_answer")
def judge_answer(question: str, answer: str, context_texts: list) -> dict:
    """Judge one answer against its retrieved context chunks."""
    context = "\n\n".join(
        f"[{i + 1}] {text[:1500]}" for i, text in enumerate(context_texts)
    )
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=question, context=context, answer=answer
    )
    last_error = None
    for client in _get_clients():
        # Two attempts per client: a fresh completion usually parses fine
        # after an empty/unparsable judge response (eval run 2, q03).
        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                    max_tokens=512,
                )
                return _parse_verdict(resp.choices[0].message.content)
            except Exception as e:  # noqa: BLE001 - judge failure must not kill the run
                last_error = str(e)
                logger.warning(
                    "Judge call failed (attempt %d), retrying: %s", attempt + 1, e
                )
    logger.error("Judge failed for question %r: %s", question, last_error)
    return {"faithful": None, "answers_question": None, "reason": "", "error": last_error}
