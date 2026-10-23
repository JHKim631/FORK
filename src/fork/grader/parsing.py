"""Robust JSON extraction from grader output.

Every grader prompt in this repository asks for a single fenced JSON object,
but a model may wrap it in prose, echo an example fence from the system prompt
before its real answer, or emit slightly malformed JSON. The strategy: take the
LAST fenced block (an echoed example comes first), fall back to the first
balanced {...} / [...] span, then strict json.loads, then json-repair, then
ParseError. The grader client turns a ParseError into a failed call, which the
reward's fallback policy handles (rewards/fallback.py).
"""

from __future__ import annotations

import json
import re

try:
    from json_repair import repair_json
except Exception:  # pragma: no cover - json_repair is a hard dep, guard for import order
    repair_json = None

__all__ = ["ParseError", "extract_json_str", "loads_lenient", "coerce_bool"]


class ParseError(ValueError):
    """Raised when no valid JSON object/array can be recovered from model text."""


def coerce_bool(v) -> bool:
    """Interpret a judgment value from model JSON as a boolean.

    Accepts real booleans, numbers, and common affirmative strings; anything
    else is False — conservative: an unparseable judgment never grants credit.
    Shared by the recall and precision parsers so the accepted spellings cannot
    drift between the two reward terms.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1", "met"}
    return False


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
# Reasoning-model channels (<think>...</think>) can precede the answer and carry
# their own braces / example JSON. Strip CLOSED think spans before scanning so
# _scan_balanced does not lock onto JSON inside the reasoning. An UNclosed
# <think> means the body was truncated -> the client fails it before parsing.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _last_fenced_block(text: str) -> str | None:
    """Return the inner text of the LAST ```...``` block, or None.

    We take the last block because a model may echo an example fence from the
    system prompt before emitting its real answer; the real answer comes last.
    """
    blocks = _FENCE_RE.findall(text)
    for block in reversed(blocks):
        if block.strip():
            return block.strip()
    return None


def _scan_balanced(text: str) -> str | None:
    """Extract the first balanced {...} or [...] span, ignoring braces in strings.

    Falls back gracefully: if no balanced close is found, returns from the first
    opener to the last matching-type closer so json-repair still gets a chance.
    """
    start = None
    opener = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            opener = ch
            break
    if start is None:
        return None
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    # Unbalanced: hand the tail to json-repair from the last closer of this type.
    last = text.rfind(closer)
    if last > start:
        return text[start : last + 1]
    return text[start:]


def extract_json_str(text: str) -> str:
    """Return the most likely JSON substring from raw model text.

    Prefers a fenced block; otherwise scans for a balanced object/array.
    Raises ParseError if nothing plausible is present.
    """
    if not text or not text.strip():
        raise ParseError("empty model output")
    text = _THINK_RE.sub(" ", text)

    fenced = _last_fenced_block(text)
    candidate = fenced if fenced is not None else text
    span = _scan_balanced(candidate)
    if span is None:
        # Fenced block may itself be bare JSON without braces stripped; retry raw.
        span = _scan_balanced(text)
    if span is None:
        raise ParseError("no JSON object/array found in model output")
    return span


def loads_lenient(text: str):
    """Parse JSON from model text, repairing common malformations.

    Order: extract candidate span -> json.loads -> json_repair -> ParseError.
    """
    span = extract_json_str(text)
    try:
        return json.loads(span)
    except json.JSONDecodeError:
        pass
    if repair_json is not None:
        try:
            repaired = repair_json(span, return_objects=False)
            obj = json.loads(repaired)
            # json_repair returns "" for hopeless input; treat as failure.
            if obj == "" and span.strip() not in ('""', "''"):
                raise ParseError("json_repair could not recover a value")
            return obj
        except (json.JSONDecodeError, ValueError) as exc:
            raise ParseError(f"json_repair failed: {exc}") from exc
    raise ParseError("invalid JSON and json_repair unavailable")
