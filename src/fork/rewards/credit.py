"""Credit-based necessity, part 1: matching rubrics to units and turning the
matches into necessity credits (paper §3.2, Eq. 2; Figure 5).

After the recall call, the grader is shown each rubric that the recall grader
judged MET, with that grader's reason, and names the units that are the basis
of the verdict — the "Rubrics to Units Matching" call. Each matched rubric
carries one credit, divided equally among the K_j units matched to it; a
unit's credit c_i is the sum of its shares, so a point spread over many units
is discounted and a unit that serves several rubrics accumulates credit.

A rubric the grader attributes to the response as a whole ("whole-response":
tone, organization, what the response refrains from doing) credits every unit
by the same share. That share cannot separate units, so it is kept apart as
`floor` and the credit shown to the threshold grader excludes it.

    parse_coverage_verdicts   the recall grader's raw JSON -> {rubric: (met, reason)}
    render_verdict_feed       the matching prompt's {rubric_block}: the MET rubrics + reasons
    parse_matching            the matching call's JSON -> one fail-closed entry per MET rubric
    matching_counters         parse-health counters for the step log
    credit_sets               {rubric: the units it credits}
    unit_credits              per-unit credit, as exact fractions

Fail-closed contract of the matching parser: an absent or malformed entry
credits nothing (`missing` / `malformed`); an explicit [] credits nothing and
is counted as a disagreement with the recall verdict (`empty`);
"whole-response" — also as the sole element of a list — credits every unit;
indices outside 1..N are dropped; entries for rubrics not marked MET are
dropped and counted (`extra_entries`). Locally malformed bodies get a strict
per-entry salvage; a body with no JSON object at all raises ParseError, so the
call fails closed in the client and the group-median fallback applies.

Arithmetic is exact (fractions.Fraction): a credit is a sum of unit fractions
compared with a threshold in a two-decimal grain, so ties must not depend on
float rounding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction

from ..grader.parsing import ParseError, coerce_bool, loads_lenient

__all__ = [
    "WHOLE",
    "parse_coverage_verdicts",
    "render_verdict_feed",
    "parse_matching",
    "matching_counters",
    "UnitCredit",
    "credit_sets",
    "unit_credits",
]

WHOLE = "whole-response"
_ENTRY_RE = re.compile(r'"(\d{1,3})"\s*:\s*(?=\{)')
_STRICT = json.JSONDecoder()
_CREDITED = ("list", "whole")


def parse_coverage_verdicts(raw: str, n_criteria: int) -> dict[int, tuple[bool, str]]:
    """{rubric number: (met, grader explanation)} from the recall grader's
    JSON; rubrics the grader returned no usable entry for are absent (the
    feed omits them and the caller counts them). Never raises: an unparseable
    body yields {} — the recall call itself already failed closed upstream
    in that case."""
    try:
        obj = loads_lenient(raw)
    except Exception:
        return {}
    out: dict[int, tuple[bool, str]] = {}
    if isinstance(obj, dict):
        for j in range(1, n_criteria + 1):
            v = obj.get(str(j))
            if isinstance(v, dict) and "criteria_met" in v:
                out[j] = (coerce_bool(v["criteria_met"]), str(v.get("explanation", "")).strip())
    return out


def render_verdict_feed(texts: list[str], verdicts: dict[int, tuple[bool, str]],
                        include_unmet: bool = False) -> str:
    """The matching prompt's {rubric_block}: `N. [MET] <criterion>` + `   reason:
    <explanation>` per rubric, ORIGINAL rubric numbering, rubrics without a
    verdict omitted. Met-only by default; include_unmet renders the NOT MET
    rubrics as context."""
    lines = []
    for j, text in enumerate(texts, start=1):
        if j not in verdicts:
            continue
        met, why = verdicts[j]
        if not met and not include_unmet:
            continue
        lines.append(f"{j}. [{'MET' if met else 'NOT MET'}] {text}\n   reason: {why}")
    return "\n".join(lines)


def _entry(v, n: int) -> dict | None:
    """One rubric entry -> {status, units, dropped}; None if malformed."""
    if not isinstance(v, dict) or "units" not in v:
        return None
    u = v["units"]
    if isinstance(u, str):
        if u.strip().lower() == WHOLE:
            return {"status": "whole", "units": list(range(1, n + 1)), "dropped": 0}
        return None
    if not isinstance(u, list):
        return None
    if not u:
        return {"status": "empty", "units": [], "dropped": 0}
    if len(u) == 1 and isinstance(u[0], str) and u[0].strip().lower() == WHOLE:
        # observed model variant (the token wrapped in the array): unambiguous, accepted, counted
        return {"status": "whole", "units": list(range(1, n + 1)), "dropped": 0, "token_in_list": True}
    kept, dropped = set(), 0
    for x in u:
        try:
            i = int(x)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if 1 <= i <= n:
            kept.add(i)
        else:
            dropped += 1
    if not kept:
        return None  # every cited index invalid: not an explicit none
    return {"status": "list", "units": sorted(kept), "dropped": dropped}


def parse_matching(raw: str, met: list[int], n: int) -> dict:
    """{"entries": {j: entry for every MET rubric j}, "extra_entries": k}.
    Raises ParseError (-> the call fails closed in the client) when no JSON
    object can be recovered at all; otherwise every met rubric gets an entry,
    `missing` for the ones the body lacks."""
    obj = loads_lenient(raw)
    if not isinstance(obj, dict):
        raise ParseError(f"matching must be a JSON object, got {type(obj).__name__}")
    met_set = set(met)
    entries: dict[int, dict] = {}
    for j in met_set:
        v = obj.get(str(j))
        if v is None:
            continue
        entries[j] = _entry(v, n) or {"status": "malformed", "units": [], "dropped": 0}
    if len(entries) < len(met_set):  # strict per-entry salvage of keys the object parse lost
        for m in _ENTRY_RE.finditer(raw):
            j = int(m.group(1))
            if j not in met_set or j in entries:
                continue
            try:
                v, _ = _STRICT.raw_decode(raw, m.end())
            except json.JSONDecodeError:
                continue
            e = _entry(v, n)
            if e:
                entries[j] = e
    for j in met_set - set(entries):
        entries[j] = {"status": "missing", "units": [], "dropped": 0}
    extra = sum(1 for k in obj if not (str(k).isdigit() and int(k) in met_set))
    return {"entries": entries, "extra_entries": extra}


def matching_counters(parsed: dict) -> dict[str, int]:
    """Parse-health counters over one parsed matching call (the step-line
    reads): `missing`, `malformed`, `disagreement` (= explicit []),
    `whole_response`, `whole_in_list`, `out_of_range` (dropped indices),
    `extra_entries`, `n_met`."""
    entries = parsed["entries"]
    return {
        "n_met": len(entries),
        "missing": sum(1 for e in entries.values() if e["status"] == "missing"),
        "malformed": sum(1 for e in entries.values() if e["status"] == "malformed"),
        "disagreement": sum(1 for e in entries.values() if e["status"] == "empty"),
        "whole_response": sum(1 for e in entries.values() if e["status"] == "whole"),
        "whole_in_list": sum(1 for e in entries.values() if e.get("token_in_list")),
        "out_of_range": sum(int(e.get("dropped", 0)) for e in entries.values()),
        "extra_entries": int(parsed.get("extra_entries", 0)),
    }


@dataclass(frozen=True)
class UnitCredit:
    credit: Fraction  # c_i: Σ 1/K_j over the specific (K_j < N) rubrics crediting the unit
    floor: Fraction   # Σ 1/K_j over the whole-response (K_j == N) rubrics — the same for every unit
    min_k: int        # smallest K among the unit's specific credits (0 = none)
    n_spec: int       # number of specific rubrics crediting the unit


def credit_sets(entries: dict, n_units: int) -> dict[int, frozenset[int]]:
    """{rubric number: credited unit indices}, rubrics that credit nothing omitted.
    Entries are parse_matching's ({status, units}; only `list` and `whole`
    credit) or bare iterables of unit indices. Indices outside 1..N are ignored
    and repeats collapse, so K_j counts distinct real units."""
    out: dict[int, frozenset[int]] = {}
    for j, v in entries.items():
        if isinstance(v, dict):
            if v.get("status") not in _CREDITED:
                continue
            idx = v.get("units") or []
        else:
            idx = v
        units = frozenset(int(i) for i in idx if 1 <= int(i) <= n_units)
        if units:
            out[int(j)] = units
    return out


def unit_credits(entries: dict, n_units: int) -> list[UnitCredit]:
    """Per-unit credit / floor / min-K / n_spec (index i-1 for unit i)."""
    credits = credit_sets(entries, n_units)
    floor = sum((Fraction(1, len(u)) for u in credits.values() if len(u) == n_units), Fraction(0))
    s = [Fraction(0)] * n_units
    min_k = [0] * n_units
    n_spec = [0] * n_units
    for u in credits.values():
        k = len(u)
        if k == n_units:
            continue
        for i in u:
            s[i - 1] += Fraction(1, k)
            n_spec[i - 1] += 1
            if min_k[i - 1] == 0 or k < min_k[i - 1]:
                min_k[i - 1] = k
    return [UnitCredit(s[i], floor, min_k[i], n_spec[i]) for i in range(n_units)]
