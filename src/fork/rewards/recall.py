"""Rubric recall — the standard rubric-coverage reward (paper §3.1, Eq. 1).

    Rec = [ sum_{w>0} w*s + sum_{w<0} |w|*(1-s) ] / sum_j |w|

where s_j in {0,1} is the grader's criteria_met judgment for rubric j. This is
the weighted fraction of satisfied rubrics; a negatively weighted (pitfall)
rubric earns its credit when it is NOT met. This module only parses the
grader's JSON and aggregates; the grader call lives in grader/client.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.schemas import Rubric
from ..grader.parsing import coerce_bool, loads_lenient

__all__ = ["RecallResult", "parse_coverage_judgments", "aggregate_recall", "recall_from_text"]


@dataclass(frozen=True)
class RecallResult:
    score: float
    n_rubrics: int
    n_met: int
    n_missing: int  # rubric indices the grader did not return (zero credit either sign)


def parse_coverage_judgments(raw: str, n_rubrics: int) -> dict[str, bool]:
    """Parse the grader's JSON into {rubric_index_str: criteria_met_bool}.

    Accepts the shape {"1": {"criteria_met": bool, "explanation": str}} and
    also tolerates a bare {"1": true} shape. Missing/garbage entries are
    simply absent from the returned dict; the aggregator gives them zero credit.
    """
    obj = loads_lenient(raw)
    if not isinstance(obj, dict):
        raise TypeError(f"coverage judgments must be a JSON object, got {type(obj).__name__}")
    out: dict[str, bool] = {}
    for idx in range(1, n_rubrics + 1):
        key = str(idx)
        if key not in obj:
            continue
        val = obj[key]
        if isinstance(val, dict):
            if "criteria_met" in val:
                out[key] = coerce_bool(val["criteria_met"])
        else:
            out[key] = coerce_bool(val)
    return out


def aggregate_recall(rubrics: list[Rubric], judgments: dict[str, bool]) -> RecallResult:
    """Weighted recall in [0, 1].

    A missing judgment earns ZERO numerator credit regardless of weight sign:
    for w > 0 that equals not-met; for w < 0 it does NOT pay the |w| avoidance
    credit (an absent judgment must never manufacture reward). |w| stays in
    the denominator either way.
    """
    num = 0.0
    den = 0.0
    n_met = 0
    n_missing = 0
    for idx, r in enumerate(rubrics, start=1):
        w = r.weight
        den += abs(w)
        key = str(idx)
        if key not in judgments:
            n_missing += 1
            continue
        g = judgments[key]
        if g:
            n_met += 1
        if w > 0:
            num += w * (1.0 if g else 0.0)
        elif w < 0:
            num += abs(w) * (0.0 if g else 1.0)
    score = num / den if den > 0 else 0.0
    return RecallResult(
        score=score, n_rubrics=len(rubrics), n_met=n_met, n_missing=n_missing
    )


def recall_from_text(rubrics: list[Rubric], raw: str) -> RecallResult:
    """Convenience: parse grader text and aggregate in one step."""
    judgments = parse_coverage_judgments(raw, len(rubrics))
    return aggregate_recall(rubrics, judgments)
