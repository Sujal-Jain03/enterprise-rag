"""Guardrails: input + output safety checks for the RAG pipeline.

Layers (defense in depth):
  1. Length limit — blocks context-flooding attempts.
  2. Regex injection patterns — instant, local, catches common jailbreaks.
  3. Llama Prompt Guard 2 (via Groq) — model-based injection classifier.
     If the guard call fails, we fall back to layers 1-2 only (fail open:
     a guard outage must not kill the app; the regex layer still runs).
Output:
  - Citation validation — strips [n] markers that point past the chunk list.
  - Leakage check — blocks answers that echo the system prompt or secrets.
"""
import re

import logfire
from groq import Groq

from app.config import settings


MAX_QUERY_CHARS = 2000
GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"
GUARD_THRESHOLD = 0.5

_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(the )?(previous |prior |above |your )?(instructions|prompt|rules|context)",
    r"you are now",
    r"pretend (you are|to be)",
    r"act as (if )?you (are|were)",
    r"(reveal|show|print|repeat|quote|tell) (me )?(your )?(system prompt|instructions|prompt)",
    r"what (are|is) your (system )?(instructions|prompt)",
    r"(don't|do not|dont) follow (your|the|these) (instructions|rules)",
    r"override (your )?(instructions|rules|system)",
    r"jailbreak",
    r"developer mode",
    r"sudo mode",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SECRET_RES = [
    re.compile(r"gsk_[A-Za-z0-9]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
]
_SYSTEM_PROMPT_MARKER = "You are a precise technical assistant"

_guard_client = None


def _get_guard_client():
    global _guard_client
    if _guard_client is None:
        _guard_client = Groq(api_key=settings.GROQ_API_KEY)
    return _guard_client


def _regex_check(query: str):
    """Return the matched injection pattern, or None."""
    m = _INJECTION_RE.search(query)
    return m.group(0) if m else None


def _prompt_guard_check(query: str):
    """Ask Prompt Guard 2; return (blocked, score_or_None, error_or_None)."""
    try:
        completion = _get_guard_client().chat.completions.create(
            model=GUARD_MODEL,
            messages=[{"role": "user", "content": query}],
            temperature=0,
            max_tokens=16,
        )
        raw = (completion.choices[0].message.content or "").strip()
        try:
            score = float(raw)
        except ValueError:
            logfire.warning("Prompt Guard returned non-numeric output.", raw=raw)
            return False, None, None
        return score >= GUARD_THRESHOLD, score, None
    except Exception as e:
        logfire.warning("Prompt Guard call failed, rules-only mode: {error}", error=str(e))
        return False, None, str(e)

@logfire.instrument("check_input")
def check_input(query: str):
    """Check a user query. Returns (allowed, reason, details)."""
    details = {"length_ok": True, "regex_hit": None, "guard_score": None, "guard_error": None}
    if not query or not query.strip():
        return False, "empty query", details
    if len(query) > MAX_QUERY_CHARS:
        details["length_ok"] = False
        return False, f"query too long ({len(query)} > {MAX_QUERY_CHARS} chars)", details
    hit = _regex_check(query)
    details["regex_hit"] = hit
    if hit:
        return False, f"injection pattern matched: {hit!r}", details
    blocked, score, error = _prompt_guard_check(query)
    details["guard_score"] = score
    details["guard_error"] = error
    if blocked:
        return False, f"prompt guard flagged (score {score})", details
    return True, None, details


def check_output(answer: str, num_chunks: int):
    """Validate an LLM answer. Returns (cleaned_answer, flags)."""
    flags = {"invalid_citations": [], "leaked": False}
    if not answer:
        return answer, flags
    # Leakage: system-prompt echo or a secret-looking string.
    if _SYSTEM_PROMPT_MARKER in answer or any(rx.search(answer) for rx in _SECRET_RES):
        flags["leaked"] = True
        logfire.warning("Output guard: possible leak in answer.")
        return answer, flags
    # Citations must point at real chunks; strip the ones that don't.
    def _fix(m):
        n = int(m.group(1))
        if 1 <= n <= num_chunks:
            return m.group(0)
        flags["invalid_citations"].append(n)
        return ""
    cleaned = _CITATION_RE.sub(_fix, answer)
    if flags["invalid_citations"]:
        logfire.info("Output guard: stripped invalid citations.", invalid=flags["invalid_citations"])
    return cleaned, flags
