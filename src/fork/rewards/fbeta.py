"""The F-beta aggregation of rubric recall and response precision (paper §3.3,
Eq. 6):

    F_beta(Prec, Rec) = (1 + beta^2) * Prec * Rec / (beta^2 * Prec + Rec)

beta = 1 is the harmonic mean; beta > 1 gives more weight to recall. Inputs are
clamped to [0, 1], and when Prec = Rec = 0 the score is 0 — the limit, and no
division by zero on the reward path. The harmonic mean is non-compensatory:
more coverage cannot make up for arbitrarily low precision, nor the reverse.
"""

from __future__ import annotations

__all__ = ["f_beta"]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def f_beta(recall: float, precision: float, beta: float) -> float:
    """F-beta score. beta > 1 weights recall more heavily. Returns 0 if P=R=0."""
    if beta <= 0:
        raise ValueError("beta must be > 0")
    r = _clamp01(recall)
    p = _clamp01(precision)
    b2 = beta * beta
    den = b2 * p + r
    if den <= 0.0:
        return 0.0
    return (1.0 + b2) * p * r / den
