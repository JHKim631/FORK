"""Response precision (paper §3.2, Eq. 5): the fraction of a response's units
that are necessary and factually correct,

    Prec = (1/n) Σ_i N_i C_i

over the n units the grader judged. This module holds what the two precision
measurements share — the result type, the unit sample, the call chunking — and
the FORK composition itself; the Generic Criteria composition is in generic.py.

Sampling: a response's units are all judged up to MAX_UNITS; above that a
seeded uniform subsample of MAX_UNITS units is judged and precision is an
unbiased estimate over the sample (the true unit count m is still reported).
The seed is a hash of the unit texts, so the same response always yields the
same sample while any change to the response reshuffles it — padding cannot
be steered into unsampled slots.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from dataclasses import dataclass, field

from .verdicts import CORRECT, CORRECT_LENIENT

__all__ = [
    "PrecisionResult",
    "MAX_UNITS",
    "LABEL_CHUNK",
    "HEDGE_RE",
    "select_label_sample",
    "chunk_ranges",
    "compose_precision",
]

# The sample cap: every unit is judged up to here, in chunks of LABEL_CHUNK
# per call for the Generic Criteria labeler (64 units per call ≈ 3.1k output
# tokens, inside the label call's token cap); above it the same seeded
# subsample applies to every chain, renumbered 1..cap.
MAX_UNITS = 256
LABEL_CHUNK = 64
# Telemetry only: a judged unit "carries a hedge cue" when it matches this —
# the cheap watch on the hedging escape the correctness prompt is written
# against.
HEDGE_RE = re.compile(r"\b(?:may|might|can|could|often|some|in some cases|generally)\b",
                      re.IGNORECASE)
_UNIT_CITE_RE = re.compile(r"\b\d{1,3}\b")


@dataclass(frozen=True)
class PrecisionResult:
    score: float
    m: int  # TRUE unit count of the response (uncapped)
    n_valid: int  # judged units passing the composition
    n_missing: int  # judged units the grader did not return a valid verdict for (treated as not good)
    n_labeled: int = 0  # units actually judged (== m unless subsampled at MAX_UNITS)
    # The chain's telemetry (unit counts, threshold reads, verdict shares, the
    # registered recomputations) — step-log material, never the trained signal.
    telemetry: dict = field(default_factory=dict)


def _unit_texts(units: list) -> list[str]:
    """Unit texts for either unit shape: plain strings or the unitizer's dicts
    ({text, section, intro, line})."""
    return [u["text"] if isinstance(u, dict) else u for u in units]


def select_label_sample(units: list, cap: int = MAX_UNITS) -> list:
    """The units to send to the grader: all of them when m <= cap, else a
    seeded uniform subsample of cap units (original order preserved). Accepts
    string units or unit dicts; element type is preserved."""
    if len(units) <= cap:
        return list(units)
    texts = _unit_texts(units)
    seed = int.from_bytes(hashlib.sha256("\x00".join(texts).encode()).digest()[:8], "big")
    keep = sorted(random.Random(seed).sample(range(len(units)), cap))
    return [units[i] for i in keep]


def chunk_ranges(n: int, size: int = LABEL_CHUNK) -> list[tuple[int, int]]:
    """1-based inclusive (lo, hi) call ranges covering n units: 130 ->
    [(1, 64), (65, 128), (129, 130)]."""
    return [(lo, min(lo + size - 1, n)) for lo in range(1, n + 1, size)]


def compose_precision(n_labeled: int, above_threshold, verdicts: dict[int, dict],
                      correctness: dict[int, dict], units: list | None = None,
                      total_units: int | None = None, telemetry: dict | None = None
                      ) -> PrecisionResult:
    """The FORK composition (strict):

        good_i = (c_i ≥ τ) ∧ (verdict_i = keep) ∧ (correctness_i ∈ {correct, na})
        Prec   = Σ good_i / n_labeled            (n_labeled = ALL sampled units)

    `above_threshold` are the (1-based) units at or above the response's
    threshold — the only units the two verdict calls judged; every other unit
    is unnecessary by the threshold, with no verdict expected. A unit above
    the threshold with a missing or invalid verdict on EITHER axis is not good
    (fail-closed); zero units ⇒ Prec = 0; no unit above the threshold ⇒
    Prec = 0 with `short_circuit: none_above_threshold` (the client skips both
    calls). `units` (the sampled unit dicts) feeds the hedge count. The three
    recomputations `p_lenient` (uncertain counted correct), `p_necessity_only`
    (correctness masked) and `p_correctness_only` (necessity masked) ride the
    telemetry as free reads. `n_missing` = units above the threshold lacking a
    valid verdict on at least one axis."""
    m = n_labeled if total_units is None else total_units
    tele = dict(telemetry or {})
    if n_labeled == 0:
        tele.update({"short_circuit": "zero_units", "n_above_threshold": 0,
                     "n_below_threshold": 0, "below_threshold_frac": None})
        return PrecisionResult(score=0.0, m=m, n_valid=0, n_missing=0, n_labeled=0, telemetry=tele)
    nec = {int(i) for i in above_threshold if 1 <= int(i) <= n_labeled}
    counts: Counter = Counter()
    good = good_len = good_nec = good_cor = 0
    for i in sorted(nec):
        ne, ce = verdicts.get(i), correctness.get(i)
        nv = ne["verdict"] if ne else None
        cv = ce["verdict"] if ce else None
        if nv is None:
            counts["necessity_missing"] += 1
        if cv is None:
            counts["correctness_missing"] += 1
        if nv is None or cv is None:
            counts["either_missing"] += 1
        if nv == "remove":
            counts["n_remove"] += 1
            if not _UNIT_CITE_RE.search(ne.get("reasoning") or ""):
                counts["n_remove_uncited"] += 1
        elif nv == "keep":
            counts["n_keep"] += 1
        if cv is not None:
            counts[f"n_{cv}"] += 1
        if nv == "remove" and cv == "incorrect":
            counts["n_both_axes"] += 1
        keep = nv == "keep"
        good += keep and cv in CORRECT
        good_len += keep and cv in CORRECT_LENIENT
        good_nec += keep
        good_cor += cv in CORRECT
        if units is not None:
            u = units[i - 1]
            text = u["text"] if isinstance(u, dict) else str(u)
            if HEDGE_RE.search(text):
                counts["n_hedge"] += 1
    n_below = n_labeled - len(nec)
    tele.update({
        "short_circuit": None if nec else "none_above_threshold",
        "n_above_threshold": len(nec),
        "n_below_threshold": n_below, "below_threshold_frac": n_below / n_labeled,
        "necessity_missing": counts["necessity_missing"],
        "correctness_missing": counts["correctness_missing"],
        "n_remove": counts["n_remove"], "n_remove_uncited": counts["n_remove_uncited"],
        "n_keep": counts["n_keep"],
        "n_correct": counts["n_correct"], "n_incorrect": counts["n_incorrect"],
        "n_uncertain": counts["n_uncertain"], "n_na": counts["n_na"],
        "n_both_axes": counts["n_both_axes"], "n_hedge": counts["n_hedge"],
        "p_lenient": good_len / n_labeled,
        "p_necessity_only": good_nec / n_labeled,
        "p_correctness_only": good_cor / n_labeled,
    })
    return PrecisionResult(score=good / n_labeled, m=m, n_valid=good,
                           n_missing=counts["either_missing"], n_labeled=n_labeled, telemetry=tele)
