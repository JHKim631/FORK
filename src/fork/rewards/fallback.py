"""Per-GRPO-group reward assembly with the grader-failure fallback policy.

A grader call can fail after the client's own retries (a transport error, a
truncated body, no JSON to parse). Handing such a rollout a gate-free reward —
"precision failed, so reward recall alone" — would make breaking the precision
grader a learnable escape from the F-beta gate, and the responses that break
it are exactly the long, degenerate ones the gate exists to police. Instead,
per rollout within one GRPO group (all rollouts of one prompt):

    recall failed                 -> reward = 0
    precision failed, recall ok   -> reward = combine(recall, P~)
    both ok                       -> reward = combine(recall, precision)

where P~ is the median of the precision values that DID verify in the same
group and `combine` is the run's aggregator (rewards/fbeta.f_beta at the
run's beta). Whole-group precision failure -> P~ = 0, so those rollouts score
0. Either way P~ is constant within the group, so the precision-side advantage
of a failed rollout is exactly zero — no gradient toward breaking the grader —
while its own recall keeps it comparable to its siblings on coverage. The
median is computed only from this group's own grader outputs, so rewards stay
deterministic given (prompt, rollouts, grader IO) and nothing leaks across
groups.

fallback_length_report is the always-on alarm that says whether the client's
retries suffice or this backstop is doing load-bearing work: fallbacks
concentrated in the longest responses mean the policy is learning to break
the grader.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median

__all__ = ["RolloutReward", "assemble_group_rewards", "fallback_length_report"]


@dataclass(frozen=True)
class RolloutReward:
    reward: float
    recall: float | None  # None = the recall call failed
    precision: float | None  # None = the precision path failed
    precision_used: float | None  # the P that entered the combiner: own, or the group median
    recall_failed: bool
    precision_failed: bool
    used_group_median: bool


def assemble_group_rewards(
    recall_scores: list[float | None],
    precision_scores: list[float | None],
    combine: Callable[[float, float], float],
) -> list[RolloutReward]:
    """Final rewards for ONE group. None marks a failed call path (after the
    client's own retries); a computed precision of 0.0 (e.g. a response with
    no units) is a real measurement, not a failure, and participates in the
    group median. `combine(recall, precision)` is the run's aggregator."""
    if len(recall_scores) != len(precision_scores):
        raise ValueError(
            f"group size mismatch: {len(recall_scores)} recall vs {len(precision_scores)} precision")
    verified = [p for p in precision_scores if p is not None]
    p_med = float(median(verified)) if verified else 0.0
    out: list[RolloutReward] = []
    for rec, p in zip(recall_scores, precision_scores):
        if rec is None:
            out.append(RolloutReward(0.0, None, p, None, True, p is None, False))
        elif p is None:
            out.append(RolloutReward(combine(rec, p_med), rec, None, p_med, False, True, True))
        else:
            out.append(RolloutReward(combine(rec, p), rec, p, p, False, False, False))
    return out


def fallback_length_report(
    rewards: list[RolloutReward], response_lengths: list[int], n_bins: int = 4
) -> dict:
    """Fallback rate by response-length bin, for per-step logging.

    Pass every rollout of the step (groups concatenated) with its response
    length in chars. Bins are contiguous quantiles of length. The signal to
    alarm on: precision_failed_rate rising with length, or overall rates > 1%.
    """
    if len(rewards) != len(response_lengths):
        raise ValueError(f"{len(rewards)} rewards vs {len(response_lengths)} lengths")
    n = len(rewards)
    overall = {
        "n": n,
        "recall_failed_rate": sum(r.recall_failed for r in rewards) / n if n else 0.0,
        "precision_failed_rate": sum(r.precision_failed for r in rewards) / n if n else 0.0,
        "bins": [],
    }
    if n == 0:
        return overall
    order = sorted(range(n), key=lambda i: response_lengths[i])
    for b in range(n_bins):
        idx = order[b * n // n_bins : (b + 1) * n // n_bins]
        if not idx:
            continue
        overall["bins"].append(
            {
                "len_min": response_lengths[idx[0]],
                "len_max": response_lengths[idx[-1]],
                "n": len(idx),
                "recall_failed_rate": sum(rewards[i].recall_failed for i in idx) / len(idx),
                "precision_failed_rate": sum(rewards[i].precision_failed for i in idx) / len(idx),
            }
        )
    return overall
