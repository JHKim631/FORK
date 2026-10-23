"""The Generic Criteria precision (paper §4.1 and App. A.3; Figure 14): the
ablation that replaces credit-based necessity with a prompting-based judgment.
One grader call labels every unit against two criteria, `correct` and
`necessary`, and a unit is good when both hold:

    good_i = correct_i ∧ necessary_i,   Prec = Σ good_i / n_labeled

The same prompt is also the evaluation's precision grader (App. E.2).

The call is chunked: every unit is rendered in every call, grouped under its
[section] lines, and {judge_range} assigns the slice this call judges, so
cross-chunk redundancy stays visible to the grader. Entries are parsed by
their 1-based GLOBAL unit numbers, and entries lost to a local JSON slip are
recovered by a strict per-entry re-read (fill-only). A unit without an entry
counts as not good.
"""

from __future__ import annotations

import json
import re

from ..grader.parsing import coerce_bool, loads_lenient
from .precision import PrecisionResult

__all__ = ["CRITERIA", "parse_unit_labels", "parse_unit_labels_salvage", "generic_precision"]

# The per-unit criteria the labeler answers, hence the keys the parser reads
# and the entries the salvage pass accepts.
CRITERIA = ("correct", "necessary")

# Anchors a per-unit entry ("13": {...}) for the salvage pass below.
_ENTRY_RE = re.compile(r'"(\d{1,3})"\s*:\s*(?=\{)')
_STRICT_JSON = json.JSONDecoder()


def parse_unit_labels(raw: str, m: int, first: int = 1,
                      keys: tuple[str, ...] = CRITERIA) -> dict[int, dict[str, bool]]:
    """Parse the labeler's JSON into {unit_index: {<keys>}}.

    Keys are 1-based to match render_units_block. Only indices in
    [first, first + m - 1] are kept and returned as-is (global indices) — each
    chunk of a chunked call parses its own global range, so the merged dict
    lines up with the full rendered unit list.
    """
    obj = loads_lenient(raw)
    if not isinstance(obj, dict):
        raise TypeError(f"unit labels must be a JSON object, got {type(obj).__name__}")
    out: dict[int, dict[str, bool]] = {}
    for i in range(first, first + m):
        val = obj.get(str(i))
        if not isinstance(val, dict):
            continue
        out[i] = {k: coerce_bool(val.get(k)) for k in keys}
    return out


def parse_unit_labels_salvage(raw: str, m: int, first: int = 1,
                              keys: tuple[str, ...] = CRITERIA) -> dict[int, dict[str, bool]]:
    """parse_unit_labels + per-entry recovery of locally malformed bodies.

    A grader occasionally breaks the JSON mid-body — a brace slip, a duplicated
    or skipped unit number — after which the lenient object parse keeps only a
    prefix and the rest of the judgments, present verbatim in the text, would
    count as missing: a length-correlated deflation of precision. This pass
    fills ONLY the missing indices, reading each `"N": {...}` entry with a
    strict decoder — an entry either parses cleanly with every key or stays
    missing. It never overrides an entry the object parse kept, and truncated
    bodies still fail closed before any parsing, so this salvages local syntax
    damage in COMPLETE bodies only.
    """
    out = parse_unit_labels(raw, m, first, keys)
    if len(out) == m:
        return out
    for mt in _ENTRY_RE.finditer(raw):
        i = int(mt.group(1))
        if not (first <= i < first + m) or i in out:
            continue  # out of judged range, or already kept (first occurrence wins)
        try:
            obj, _ = _STRICT_JSON.raw_decode(raw, mt.end())
        except json.JSONDecodeError:
            continue  # the damaged entry itself: stays missing -> not good
        if isinstance(obj, dict) and all(k in obj for k in keys):
            out[i] = {k: coerce_bool(obj[k]) for k in keys}
    return out


def generic_precision(labeled_units: list, labels: dict[int, dict],
                      total_units: int | None = None) -> PrecisionResult:
    """Prec = (# judged units with correct AND necessary) / n_labeled.

    `labeled_units` is what went to the labeler (select_label_sample's output;
    indices in `labels` are 1-based positions in THAT list). `total_units` is
    the response's true m when the labeler saw a subsample. Missing labels ->
    not good: an unlabeled unit earns no precision credit.
    """
    n_labeled = len(labeled_units)
    m = n_labeled if total_units is None else total_units
    if n_labeled == 0:
        return PrecisionResult(score=0.0, m=m, n_valid=0, n_missing=0, n_labeled=0)
    n_valid = 0
    n_missing = 0
    for i in range(1, n_labeled + 1):
        lab = labels.get(i)
        if lab is None:
            n_missing += 1
            continue
        if lab["correct"] and lab["necessary"]:
            n_valid += 1
    return PrecisionResult(score=n_valid / n_labeled, m=m, n_valid=n_valid,
                           n_missing=n_missing, n_labeled=n_labeled)
