"""The per-unit verdict calls' shared parser.

The necessity-threshold call and the factual-correctness call both answer
with a JSON object keyed by unit number, each value {"reasoning", "verdict"},
over an explicit list of assigned unit numbers (every unit is shown; only the
assigned ones are judged). This module holds the two vocabularies and the one
parser both calls use.

Parsing is fail-closed: a unit without a valid entry is `missing` and never
counts as good. Two mitigations for observed output slips are built in: a
bare string that IS a verdict token is accepted as a flagged entry with empty
reasoning (the verdict carries the signal), and entries lost to a local JSON
slip are recovered by a strict per-entry re-read of the body (fill-only: an
entry the object parse kept is never overridden). A body with no JSON object
at all raises ParseError, so the call fails closed in the client.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from ..grader.parsing import ParseError, loads_lenient

__all__ = [
    "CHUNK",
    "NECESSITY_VOCAB",
    "CORRECTNESS_VOCAB",
    "CORRECT",
    "CORRECT_LENIENT",
    "chunk_index_list",
    "parse_verdicts_listed",
]

# Units judged per call for both instruments, all units rendered every call.
CHUNK = 64

# In-prompt verdict vocabularies (case-insensitive; values are the canonical
# spellings the composition and telemetry key on).
# Necessity threshold call: "remove" marks a unit above the threshold that only
# repeats an earlier unit above the threshold.
NECESSITY_VOCAB = {"keep": "keep", "remove": "remove"}
# Factual-correctness call: `uncertain` is the one token for a claim that is
# neither contradicted nor confirmed (an unverifiable specific, an
# overstatement); `unsupported` and `unknown` are accepted as aliases of it so
# a stray token lands in the class instead of counting as missing. `na` is for
# a unit that makes no factual claim.
CORRECTNESS_VOCAB = {"correct": "correct", "incorrect": "incorrect",
                     "uncertain": "uncertain", "unsupported": "uncertain", "unknown": "uncertain",
                     "na": "na", "n/a": "na"}
# The composition is strict: `uncertain` is NOT correct. The lenient set is a
# free telemetry read, never the trained signal.
CORRECT = frozenset({"correct", "na"})
CORRECT_LENIENT = frozenset({"correct", "uncertain", "na"})

_ENTRY_RE = re.compile(r'"(\d{1,3})"\s*:\s*(?=\{)')
_STRICT = json.JSONDecoder()


def _entry(v, vocab: dict[str, str]) -> tuple[dict | None, Counter]:
    """One unit entry -> ({"reasoning", "verdict"}, flags) or (None, flags).
    A bare string that IS a verdict token is accepted as a flagged fallback
    (`bare_entry`); a bare reasoning string has no verdict and stays malformed."""
    flags: Counter = Counter()
    if v is None:
        flags["absent"] += 1
        return None, flags
    if isinstance(v, str):
        s = vocab.get(v.strip().lower())
        if s is not None:
            flags["bare_entry"] += 1
            return {"reasoning": "", "verdict": s}, flags
        flags["malformed"] += 1
        return None, flags
    if not isinstance(v, dict):
        flags["malformed"] += 1
        return None, flags
    s = vocab.get(str(v.get("verdict", "")).strip().lower())
    if s is None:
        flags["bad_verdict"] += 1
        return None, flags
    reasoning = v.get("reasoning")
    if not (isinstance(reasoning, str) and reasoning.strip()):
        flags["no_reasoning"] += 1
        reasoning = "" if reasoning is None else str(reasoning)
    return {"reasoning": reasoning.strip(), "verdict": s}, flags


def chunk_index_list(indices, size: int = CHUNK) -> list[list[int]]:
    """Consecutive slices of a sorted, de-duplicated index list, at most
    `size` per chunk."""
    idx = sorted({int(i) for i in indices})
    return [idx[k:k + size] for k in range(0, len(idx), size)]


def parse_verdicts_listed(raw: str, assigned, vocab: dict[str, str]) -> tuple[dict[int, dict], dict]:
    """{unit: entry} for the ASSIGNED unit numbers + per-cause counters
    (`absent`, `malformed`, `bad_verdict`, `no_reasoning`, `bare_entry`,
    `out_of_range` = keys outside the list, `salvaged`, `missing` = assigned
    units without a valid entry). A body with no JSON object raises ParseError
    (the call fails closed in the client); the bail check (fewer than half the
    assigned units valid) is the caller's — this parser never raises on thin
    bodies, so a bailed body still yields its entries for the audit trail."""
    want = sorted({int(i) for i in assigned})
    want_set = set(want)
    flags: Counter = Counter()
    obj = loads_lenient(raw)
    if not isinstance(obj, dict):
        raise ParseError(f"verdicts must be a JSON object, got {type(obj).__name__}")
    for key in obj:
        try:
            i = int(key)
        except (TypeError, ValueError):
            continue
        if i not in want_set:
            flags["out_of_range"] += 1
    entries: dict[int, dict] = {}
    for i in want:
        e, f = _entry(obj.get(str(i)), vocab)
        flags.update(f)
        if e:
            entries[i] = e
    if len(entries) < len(want):
        for mt in _ENTRY_RE.finditer(raw):
            i = int(mt.group(1))
            if i not in want_set or i in entries:
                continue
            try:
                obj_i, _ = _STRICT.raw_decode(raw, mt.end())
            except json.JSONDecodeError:
                continue
            e, _f = _entry(obj_i, vocab)
            if e:
                entries[i] = e
                flags["salvaged"] += 1
    flags["missing"] = len(want) - len(entries)
    return entries, dict(flags)
