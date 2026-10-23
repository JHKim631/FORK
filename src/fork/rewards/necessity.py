"""Credit-based necessity, part 2: the necessity threshold decision (paper
§3.2, Eq. 3; Figures 6–7).

Every unit is shown to the grader with its credit c_i, rendered to two
decimals (grader/templates.render_units_block_scored). The grader sets the
threshold τ for this response — the lowest credit a unit must carry to be
necessary — and, for the units at or above τ, returns one verdict each:
"keep", or "remove" for a unit that only repeats an earlier unit at or above
the threshold. A unit is necessary when c_i ≥ τ and its verdict is "keep".
The credit shares are exact fractions; every comparison is made in the shown
two-decimal grain, so a threshold equal to a shown credit keeps that unit.

Long responses: one call per run of consecutive units holding at most CHUNK
units with a positive credit (a unit at 0.00 can never be necessary, so it
never counts against the cap); every call sees all units and sets a threshold
for the whole response; the response takes the STRICTEST threshold returned
and each unit's verdict from its own call. A dropped, non-numeric or
non-positive threshold falls back to 1/3 and is counted.

    score_str              the two-decimal grain (half up, from the exact fraction)
    parse_threshold        the grader's "threshold" value -> a grain value, or why not
    necessary_at           the units whose shown credit is at or above a threshold
    chunk_by_scored        the call runs over a response's units
    parse_threshold_call   one call's body -> (threshold, verdict entries, flags)
    merge_threshold_parts  the strictest-wins merge over a response's calls
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

from ..grader.parsing import ParseError, loads_lenient
from .verdicts import CHUNK, NECESSITY_VOCAB, parse_verdicts_listed

__all__ = [
    "SCORE_PLACES",
    "THRESHOLD_FALLBACK",
    "THRESHOLD_FALLBACK_SHOWN",
    "score_str",
    "parse_threshold",
    "necessary_at",
    "chunk_by_scored",
    "parse_threshold_call",
    "merge_threshold_parts",
]

THRESHOLD_FALLBACK = Fraction(1, 3)   # a dropped, non-numeric or non-positive threshold
SCORE_PLACES = 2                      # the grain: two decimals, half up
_UNIT_CITE_RE = re.compile(r"\b\d{1,3}\b")


def score_str(x) -> str:
    """The credit grain: a non-negative exact value rendered with two decimals,
    half up, from the exact fraction — `(score 0.50)` in the units block, and
    the grain every comparison is made in (a threshold is rounded the same way
    before comparing). Floats are snapped to the nearest small rational first
    (Fraction(0.1) alone is not 1/10)."""
    q = Fraction(x).limit_denominator(1_000_000)
    if q < 0:
        raise ValueError(f"score must be non-negative, got {x!r}")
    scaled = q * 10 ** SCORE_PLACES
    n = (2 * scaled.numerator + scaled.denominator) // (2 * scaled.denominator)   # floor(scaled + 1/2)
    return f"{n // 10 ** SCORE_PLACES}.{n % 10 ** SCORE_PLACES:0{SCORE_PLACES}d}"


THRESHOLD_FALLBACK_SHOWN = Fraction(score_str(THRESHOLD_FALLBACK))   # 1/3 in the grain: 0.33


def parse_threshold(value) -> tuple[Fraction | None, str | None]:
    """The grader's "threshold" value -> (the threshold in the grain, None) or
    (None, why): `absent` (missing / null), `non_numeric` (a bool, a
    non-numeric string, a list, NaN...), `non_positive` (at or below 0.00 in
    the grain — 0.004 rounds to 0.00). Any positive value is followed as
    written, above 1 included; a numeric string ("0.25", "1/3") counts."""
    if value is None:
        return None, "absent"
    if isinstance(value, bool):
        return None, "non_numeric"
    try:
        if isinstance(value, str):
            q = Fraction(value.strip())
        elif isinstance(value, (int, float)):
            q = Fraction(value)
        else:
            return None, "non_numeric"
    except (ValueError, ZeroDivisionError, OverflowError):
        return None, "non_numeric"
    if q < 0:
        return None, "non_positive"
    t = Fraction(score_str(q))
    if t <= 0:
        return None, "non_positive"
    return t, None


def necessary_at(shown: list[str], threshold: Fraction, run=None) -> list[int]:
    """1-based indices (within `run`, default every unit) whose SHOWN credit is
    at or above the threshold — compared in the grain, so a threshold equal
    to a shown credit keeps that unit."""
    idx = range(1, len(shown) + 1) if run is None else sorted({int(i) for i in run})
    return [i for i in idx if Fraction(shown[i - 1]) >= threshold]


def chunk_by_scored(shown: list[str], size: int = CHUNK) -> list[list[int]]:
    """The call cap: runs of consecutive unit indices 1..N, each holding at
    most `size` SCORED units (a shown credit above 0.00 — a unit at 0.00 can
    never be necessary, so it never counts against the cap); every unit lands
    in exactly one run, zero-credit units riding with the run in progress;
    N = 0 -> []."""
    n = len(shown)
    if n == 0:
        return []
    runs: list[list[int]] = []
    cur: list[int] = []
    k = 0
    for i in range(1, n + 1):
        scored = Fraction(shown[i - 1]) > 0
        if scored and k == size:
            runs.append(cur)
            cur, k = [], 0
        cur.append(i)
        k += scored
    runs.append(cur)
    return runs


def parse_threshold_call(raw: str, run, shown: list[str], vocab: dict[str, str] = NECESSITY_VOCAB
                         ) -> tuple[Fraction, dict[int, dict], dict]:
    """One threshold call's body -> (threshold, entries over the units of the
    run at or above it, flags). The threshold is read first (parse_threshold;
    a dropped, non-numeric or non-positive one -> THRESHOLD_FALLBACK in the
    grain, `thr_fallback` = 1 with `thr_absent` / `thr_non_numeric` /
    `thr_non_positive` saying why); the units owed an entry = the run's units
    whose shown credit is at or above it (necessary_at); their entries follow
    parse_verdicts_listed exactly (the vocabulary, the bare-entry fallback, the
    fill-only strict salvage; `out_of_range` = keys for any other unit — below
    the threshold or outside the run; the "threshold" key is never counted;
    `missing` = units owed an entry without a valid one). `n_necessary_run`
    rides the flags for the caller's bail rule. `thr_cites_below` counts
    removes whose reasoning cites unit numbers none of which is at or above
    the threshold (an off-contract watch; the verdict is read as given). A
    body with no JSON object raises ParseError (the call fails closed)."""
    obj = loads_lenient(raw)
    if not isinstance(obj, dict):
        raise ParseError(f"threshold+verdicts must be a JSON object, got {type(obj).__name__}")
    thr, why = parse_threshold(obj.get("threshold"))
    flags: Counter = Counter()
    if thr is None:
        thr = THRESHOLD_FALLBACK_SHOWN
        flags["thr_fallback"] = 1
        flags[f"thr_{why}"] = 1
    want = necessary_at(shown, thr, run)
    entries, vflags = parse_verdicts_listed(raw, want, vocab)
    flags.update(vflags)
    flags["n_necessary_run"] = len(want)
    flags["thr_cites_below"] = 0
    n = len(shown)
    for e in entries.values():
        if e["verdict"] != "remove":
            continue
        cited = [int(c) for c in _UNIT_CITE_RE.findall(e.get("reasoning") or "") if 1 <= int(c) <= n]
        if cited and not any(Fraction(shown[c - 1]) >= thr for c in cited):
            flags["thr_cites_below"] += 1
    return thr, entries, dict(flags)


def merge_threshold_parts(parts: list[tuple[Fraction, dict[int, dict], dict]], shown: list[str]
                          ) -> tuple[Fraction, dict[int, dict], list[int], dict]:
    """The merge over the parts of one response (one per call, two for a call
    that split-retried): the response takes the STRICTEST threshold; the units
    at or above it = every unit whose shown credit is at or above it; each
    unit's verdict comes from its own part — a part judged its run at its own
    threshold, which is at most the strictest, so every unit in the merged set
    carries its part's verdict unless that part left it missing. Flags: the
    parts' counters summed (`missing` recomputed over the merged set;
    `thr_dropped_by_merge` = entries a part gave for units below the merged
    threshold), plus `n_calls` (parts), `thr_disagree` (the parts returned
    different thresholds), `thr_fallback` (the merged threshold is a
    fallback's — no part with a real threshold reached it) and
    `thr_fallback_calls`. Returns (threshold, entries, necessary, flags)."""
    if not parts:
        raise ValueError("merge_threshold_parts needs at least one part")
    thr = max(p[0] for p in parts)
    necessary = necessary_at(shown, thr)
    nec = set(necessary)
    entries: dict[int, dict] = {}
    flags: Counter = Counter()
    dropped = 0
    for t, es, f in parts:
        for i, e in es.items():
            if i in nec:
                entries[i] = e
            else:
                dropped += 1
        flags.update({k: v for k, v in f.items() if k not in ("thr_fallback", "n_necessary_run", "missing")})
    flags["missing"] = len(nec) - len(entries)
    flags["thr_dropped_by_merge"] = dropped
    flags["n_calls"] = len(parts)
    flags["thr_disagree"] = int(len({p[0] for p in parts}) > 1)
    flags["thr_fallback_calls"] = sum(int(p[2].get("thr_fallback", 0)) for p in parts)
    flags["thr_fallback"] = int(not any(p[0] == thr and not p[2].get("thr_fallback") for p in parts))
    return thr, entries, necessary, dict(flags)
