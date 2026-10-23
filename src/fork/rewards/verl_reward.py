"""verl's custom-reward entry point: the reward of every rollout of a training
batch (scripts/train.sh wires it in as custom_reward_function).

verl's BATCH reward manager calls `compute_score` once per training batch with
parallel lists over all rollouts (128 prompts × 16 rollouts = 2,048). From
verl's side the call is synchronous; internally each call runs ONE fresh
asyncio event loop with ONE fresh GraderClient — never reused across calls
(the client's semaphores and HTTP pools are loop-bound).

Reward per rollout:
    reward: recall     R = Rec                           (Standard)
    reward: fbeta      R = F_beta(Prec, Rec), precision: credit  (FORK)
                                             precision: generic (Generic Criteria)
A failed recall call scores 0; a failed precision path takes the group-median
precision of its GRPO group (rewards/fallback.py). If more than FORK_ABORT_FRAC
of a batch's rollouts needed any fallback, the step's logs are written, an
ABORT marker is left in the run directory and the batch raises: a grader
outage must stop training loudly, not feed it degenerate rewards. The marker
makes the abort STICKY — verl retries the reward function once after an
exception, and the retry refuses to run while the marker exists. train.sh
clears it at launch.

Every step writes <run dir>/reward_log.jsonl (one line of batch statistics) and
<run dir>/grader_io/step_<n>.jsonl (one record per rollout: the reward, the
scores, every grader call's raw text and token counts, the chain's telemetry)
— the audit trail every number can be traced back to.

Configuration comes from the environment, because verl's custom_reward_function
config carries no user kwargs (scripts/train.sh exports these). The
run-defining variables are REQUIRED — the launcher always exports them, so a
missing one means the Ray workers inherited a foreign environment (typically a
pre-existing `ray start` cluster) and defaulting would silently score the
wrong reward:
    FORK_REWARD         recall | fbeta                      REQUIRED
    FORK_BETA           F_beta's beta, > 0                  REQUIRED under fbeta, refused otherwise
    FORK_PRECISION      credit | generic                    REQUIRED under fbeta, refused otherwise
    FORK_DOMAIN         medical | science                   [medical]
    FORK_GRADER_CONFIG  configs/grader/<model>.yaml         [configs/grader/qwen36_35b.yaml]
    FORK_GRADER_URLS    comma-separated replica base URLs   REQUIRED
    FORK_RUN_DIR        the run's output directory          REQUIRED
    FORK_ABORT_FRAC     the hard-abort fallback fraction    [0.05]
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import yaml

# ABSOLUTE imports on purpose: verl loads this file by path
# (custom_reward_function.path), so it has no parent package and relative
# imports would raise. `fork` must be importable (pip install -e . or
# PYTHONPATH=src, which scripts/train.sh sets).
from fork.data.schemas import Rubric
from fork.grader.client import DOMAINS, CallResult, GraderClient, GraderConfig
from fork.rewards.fallback import RolloutReward, assemble_group_rewards, fallback_length_report
from fork.rewards.fbeta import f_beta
from fork.rewards.unitize import split

REWARDS = ("recall", "fbeta")
PRECISIONS = ("credit", "generic")
IO_DIR = "grader_io"

_next_steps: dict[str, int] = {}  # next step index per run_dir; seeded from disk


def _next_step(run_dir: Path) -> int:
    """Monotonic index for this run's step logs. Seeded on first use from the
    step files already in <run_dir>/grader_io, so a resumed (or verl-retried)
    process appends instead of overwriting the audit trail. It is therefore
    NOT verl's global step: line the two up by the per-line `time`."""
    key = str(run_dir)
    if key not in _next_steps:
        stems = (p.stem.removeprefix("step_") for p in (run_dir / IO_DIR).glob("step_*.jsonl"))
        nums = [int(s) for s in stems if s.isdigit()]
        _next_steps[key] = max(nums) + 1 if nums else 0
    step = _next_steps[key]
    _next_steps[key] = step + 1
    return step


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(
            f"{name} is not set in the reward worker's environment. scripts/train.sh always "
            "exports it, so the Ray workers must have inherited a stale environment "
            "(a pre-existing Ray cluster?) — `ray stop`, then relaunch via scripts/train.sh.")
    return val


def _beta() -> float:
    raw = _require_env("FORK_BETA")
    try:
        beta = float(raw)
    except ValueError:
        raise ValueError(f"FORK_BETA must be a positive number, got {raw!r}") from None
    if not beta > 0 or beta == float("inf"):
        raise ValueError(f"FORK_BETA must be a positive finite number, got {raw!r}")
    return beta


@lru_cache(maxsize=4)
def _grader_config(path: str) -> GraderConfig:
    return GraderConfig.from_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def _domain() -> str:
    dom = os.environ.get("FORK_DOMAIN", "medical").strip() or "medical"
    if dom not in DOMAINS:
        raise ValueError(f"FORK_DOMAIN must be {'|'.join(sorted(DOMAINS))}, got {dom!r}")
    return dom


def _make_client(urls: list[str], gcfg: GraderConfig) -> GraderClient:
    """Factory, fresh per batch — see the module docstring."""
    return GraderClient(urls, gcfg, domain=_domain())


def _parse_item(ground_truth: str, extra_info: dict | None, response: str) -> dict:
    gt = json.loads(ground_truth)
    prompt_id = (extra_info or {}).get("prompt_id") or gt["prompt_id"]
    return {
        "prompt_id": prompt_id,
        "prompt": gt["prompt"],
        "reference": gt.get("reference_answer") or "",
        "rubrics": [Rubric(text=r["text"], weight=float(r["weight"])) for r in gt["rubrics"]],
        "response": response if isinstance(response, str) else str(response),
    }


async def _score_one(client: GraderClient, item: dict, reward: str, precision: str | None) -> dict:
    """The grader calls of one rollout. `recall` is the recall call; under fbeta
    `units` are the response's units and `precision` the precision result —
    the label call (generic) or the composed chain result (credit, with the
    `match` / `threshold` / `correctness` calls beside it)."""
    if reward == "recall":
        cov = await client.grade_recall(item["rubrics"], item["prompt"], item["response"])
        return {"recall": cov, "units": None, "precision": None}
    units = split(item["response"])
    if precision == "generic":
        cov, lab = await asyncio.gather(
            client.grade_recall(item["rubrics"], item["prompt"], item["response"]),
            client.label(item["prompt"], units, item["rubrics"], item["reference"]),
        )
        return {"recall": cov, "units": units, "precision": lab}
    # credit: the recall call FIRST (the matching call reads its verdicts), then the chain
    cov = await client.grade_recall(item["rubrics"], item["prompt"], item["response"])
    chain = await client.precision_credit(item["prompt"], units, item["rubrics"], cov)
    return {"recall": cov, "units": units, "precision": chain["precision"],
            "match": chain["match"], "threshold": chain["threshold"], "correctness": chain["correctness"]}


async def _score_batch(items: list[dict], reward: str, precision: str | None, urls: list[str],
                       gcfg: GraderConfig) -> list[dict]:
    client = _make_client(urls, gcfg)
    try:
        return await asyncio.gather(*(_score_one(client, it, reward, precision) for it in items))
    finally:
        await client.aclose()


def _recall_score(calls: dict) -> float | None:
    cov: CallResult = calls["recall"]
    return cov.value.score if cov.ok else None


def _precision_score(calls: dict) -> float | None:
    p = calls.get("precision")
    if p is None or not p.ok:
        return None
    return p.value.score


def _assemble(items: list[dict], scored: list[dict], reward: str, beta: float | None
              ) -> list[RolloutReward]:
    """Per-rollout rewards in INPUT order; under fbeta the precision fallback is
    per GRPO group (= the rollouts of one prompt_id)."""
    if reward == "recall":
        out = []
        for calls in scored:
            rec = _recall_score(calls)
            out.append(RolloutReward(reward=rec if rec is not None else 0.0, recall=rec,
                                     precision=None, precision_used=None,
                                     recall_failed=rec is None, precision_failed=False,
                                     used_group_median=False))
        return out
    combine = lambda recall, precision: f_beta(recall, precision, beta)  # noqa: E731
    groups: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        groups.setdefault(item["prompt_id"], []).append(i)
    out: list[RolloutReward | None] = [None] * len(items)
    for idx in groups.values():
        group_rewards = assemble_group_rewards(
            [_recall_score(scored[i]) for i in idx],
            [_precision_score(scored[i]) for i in idx],
            combine=combine,
        )
        for i, rr in zip(idx, group_rewards):
            out[i] = rr
    return out  # type: ignore[return-value]


def _call_dump(r: CallResult | None) -> dict | None:
    if r is None:
        return None
    return {"ok": r.ok, "raw": r.raw, "error": r.error, "finish_reason": r.finish_reason,
            "truncation_retries": r.truncation_retries, "refusal_retries": r.refusal_retries,
            "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens}


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation (average ranks on ties); None when either
    series is constant or has fewer than 2 points."""
    n = len(xs)
    if n < 2:
        return None

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (dx * dy)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _rate(num, den):
    return round(num / den, 4) if den else None


def _credit_step_fields(items: list[dict], scored: list[dict], rewards: list[RolloutReward]) -> dict:
    """The FORK chain's step telemetry: the share of units below the threshold
    and the mean credit; the matching call's health; the verdict shares over the
    units above the threshold (removes, incorrect, uncertain, na, hedges); the
    three free recomputations; the within-group rank correlations of reward and
    precision with length, of reward with the incorrect share and the remove
    share; the threshold reads (level, histogram, fallback rate, the share of
    responses the threshold emptied, the cut it makes, the multi-call rate);
    instrument health."""
    tele = [c["precision"].value.telemetry for c in scored
            if c.get("precision") is not None and c["precision"].ok and c["precision"].value.telemetry]

    def _sum(key, sub=None):
        tot = 0
        for t in tele:
            v = t.get(sub, {}).get(key, 0) if sub else t.get(key, 0)
            tot += int(v or 0)
        return tot

    n_above = _sum("n_above_threshold")
    n_met = _sum("n_met")
    flags_sum = lambda key: (_sum(key, "necessity_flags") + _sum(key, "correctness_flags"))  # noqa: E731

    groups: dict[str, dict[str, list[float]]] = {}
    for item, rr, calls in zip(items, rewards, scored):
        g = groups.setdefault(item["prompt_id"], {"p": [], "chars_p": [], "r": [], "chars_r": [],
                                                  "inc": [], "r_inc": [], "rem": [], "r_rem": []})
        g["r"].append(rr.reward)
        g["chars_r"].append(float(len(item["response"])))
        if rr.precision is not None:
            g["p"].append(rr.precision)
            g["chars_p"].append(float(len(item["response"])))
        p = calls.get("precision")
        if p is not None and p.ok and p.value.telemetry and (p.value.telemetry.get("n_above_threshold") or 0) > 0:
            t = p.value.telemetry
            nn = t["n_above_threshold"]
            g["inc"].append((int(t.get("n_incorrect") or 0) + int(t.get("n_uncertain") or 0)) / nn)
            g["r_inc"].append(rr.reward)
            g["rem"].append(int(t.get("n_remove") or 0) / nn)
            g["r_rem"].append(rr.reward)
    sds, rho_r, rho_p, rho_inc, rho_rem = [], [], [], [], []
    for g in groups.values():
        if len(g["p"]) >= 2:
            mu = sum(g["p"]) / len(g["p"])
            sds.append((sum((x - mu) ** 2 for x in g["p"]) / (len(g["p"]) - 1)) ** 0.5)
        for xs, ys, acc in ((g["r"], g["chars_r"], rho_r), (g["p"], g["chars_p"], rho_p),
                            (g["r_inc"], g["inc"], rho_inc), (g["r_rem"], g["rem"], rho_rem)):
            if len(xs) >= 2:
                r = _spearman(xs, ys)
                if r is not None:
                    acc.append(r)

    with_scored = [t for t in tele if (t.get("n_scored") or 0) > 0]
    thr = [float(t["threshold"]) for t in with_scored if t.get("threshold") is not None]
    hist: dict[str, int] = {}
    for x in thr:
        b = f"{int(x / 0.05 + 1e-9) * 0.05:.2f}"
        hist[b] = hist.get(b, 0) + 1
    multi = [t for t in with_scored if (t.get("n_calls") or 0) > 1]

    def _median(xs):
        xs = sorted(xs)
        if not xs:
            return None
        h = len(xs) // 2
        return float(xs[h]) if len(xs) % 2 else (xs[h - 1] + xs[h]) / 2

    return {
        "below_threshold_frac": _mean([t.get("below_threshold_frac") for t in tele]),
        "mean_credit": _mean([t.get("mean_credit") for t in tele]),
        "match_whole_response_rate": _rate(_sum("whole_response", "match"), n_met),
        "match_missing": _sum("missing", "match"),
        "match_out_of_range": _sum("out_of_range", "match"),
        "match_disagreement": _sum("disagreement", "match"),
        "match_extra": _sum("extra_entries", "match"),
        "remove_rate": _rate(_sum("n_remove"), n_above),
        "remove_uncited_share": _rate(_sum("n_remove_uncited"), _sum("n_remove")),
        "incorrect_rate": _rate(_sum("n_incorrect"), n_above),
        "uncertain_frac": _rate(_sum("n_uncertain"), n_above),
        "na_frac": _rate(_sum("n_na"), n_above),
        "both_axes_rate": _rate(_sum("n_both_axes"), n_above),
        "hedge_rate": _rate(_sum("n_hedge"), n_above),
        "p_lenient": _mean([t.get("p_lenient") for t in tele]),
        "p_necessity_only": _mean([t.get("p_necessity_only") for t in tele]),
        "p_correctness_only": _mean([t.get("p_correctness_only") for t in tele]),
        "rho_reward_chars": _mean(rho_r),
        "rho_p_chars": _mean(rho_p),
        "rho_reward_incorrect": _mean(rho_inc),
        "rho_reward_remove": _mean(rho_rem),
        "none_above_threshold_rate": _rate(sum(1 for t in tele if t.get("short_circuit") == "none_above_threshold"),
                                           len(tele)),
        "necessity_missing": _sum("necessity_missing"),
        "correctness_missing": _sum("correctness_missing"),
        "bare_entry_rate": _rate(flags_sum("bare_entry"), 2 * n_above),
        "split_retries": flags_sum("split_retries"),
        "split_recovered": flags_sum("split_recovered"),
        "bail_chunks": sum(1 for c in scored if c.get("precision") is not None and not c["precision"].ok
                           and "bail:" in (c["precision"].error or "")),
        "threshold_truncations": sum(c["threshold"].truncation_retries for c in scored
                                     if c.get("threshold") is not None),
        "correctness_truncations": sum(c["correctness"].truncation_retries for c in scored
                                       if c.get("correctness") is not None),
        "p_within_group_sd": _mean(sds),
        "thr_mean": _mean(thr),
        "thr_median": _median(thr),
        "thr_hist": dict(sorted(hist.items(), key=lambda kv: float(kv[0]))),
        "thr_fallback_rate": _rate(sum(1 for t in with_scored if t.get("threshold_fallback")), len(with_scored)),
        "thr_none_rate": _rate(sum(1 for t in with_scored if t.get("thr_stage") == "none_above"), len(with_scored)),
        "necessary_per_scored": _mean([(t.get("n_necessary") or 0) / t["n_scored"] for t in with_scored]),
        "multi_call_rate": _rate(len(multi), len(with_scored)),
        "thr_disagree_rate": _rate(sum(1 for t in multi if t.get("thr_disagree")), len(multi)),
        "thr_cites_below_share": _rate(sum(int(t.get("thr_cites_below") or 0) for t in tele), _sum("n_remove")),
    }


def _write_step_logs(run_dir: Path, step: int, reward: str, beta: float | None, precision: str | None,
                     domain: str, items: list[dict], scored: list[dict], rewards: list[RolloutReward]) -> dict:
    io_dir = run_dir / IO_DIR
    io_dir.mkdir(parents=True, exist_ok=True)
    with (io_dir / f"step_{step:06d}.jsonl").open("w") as f:
        for item, calls, rr in zip(items, scored, rewards):
            p = calls.get("precision")
            rec = {
                "prompt_id": item["prompt_id"],
                "response_chars": len(item["response"]),
                "reward": rr.reward, "recall": rr.recall, "precision": rr.precision,
                "precision_used": rr.precision_used, "recall_failed": rr.recall_failed,
                "precision_failed": rr.precision_failed, "used_group_median": rr.used_group_median,
                "n_units": None if calls.get("units") is None else len(calls["units"]),
                "recall_call": _call_dump(calls["recall"]),
            }
            if precision == "generic":
                rec["label"] = _call_dump(p)
            elif precision == "credit":
                rec.update({"match": _call_dump(calls.get("match")),
                            "threshold": _call_dump(calls.get("threshold")),
                            "correctness": _call_dump(calls.get("correctness")),
                            "precision_call": _call_dump(p),
                            "telemetry": p.value.telemetry if p is not None and p.ok else None})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lengths = [len(it["response"]) for it in items]
    report = fallback_length_report(rewards, lengths)
    m_vals, cpu_vals, n_missing = [], [], 0
    for item, calls in zip(items, scored):
        p = calls.get("precision")
        if p is not None and p.ok:
            m_vals.append(float(p.value.m))
            if p.value.m > 0:
                cpu_vals.append(len(item["response"]) / p.value.m)
            n_missing += p.value.n_missing
    line = {
        "step": step, "time": round(time.time(), 3), "reward": reward, "n": len(rewards),
        **({"beta": beta, "precision": precision} if reward == "fbeta" else {}),
        "domain": domain,
        "mean_reward": _mean([r.reward for r in rewards]),
        "mean_recall": _mean([r.recall for r in rewards]),
        "mean_precision": _mean([r.precision for r in rewards]),
        "mean_response_chars": _mean([float(x) for x in lengths]),
        **({"mean_units": _mean(m_vals), "mean_chars_per_unit": _mean(cpu_vals), "n_missing": n_missing}
           if reward == "fbeta" else {}),
        **(_credit_step_fields(items, scored, rewards) if precision == "credit" else {}),
        "recall_failed": sum(r.recall_failed for r in rewards),
        "precision_failed": sum(r.precision_failed for r in rewards),
        "used_group_median": sum(r.used_group_median for r in rewards),
        "fallback_length_report": report,
    }
    with (run_dir / "reward_log.jsonl").open("a") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    _wandb_log(line)
    return line


_wandb_ready = False
_WANDB_CREDIT_KEYS = ("below_threshold_frac", "remove_rate", "incorrect_rate", "uncertain_frac", "na_frac",
                      "hedge_rate", "p_lenient", "p_necessity_only", "p_correctness_only",
                      "rho_reward_chars", "rho_p_chars", "p_within_group_sd", "none_above_threshold_rate",
                      "thr_mean", "thr_median", "thr_fallback_rate", "thr_none_rate", "necessary_per_scored")


def _wandb_log(line: dict) -> None:
    """Mirror the step's mean recall and precision (and the chain's headline
    reads) into the trainer's W&B run, beside verl's own curves. Plotted against
    `fork/step`, the audit-trail index, on its own axis. Never raises: a logging
    failure must not kill a training run."""
    global _wandb_ready
    try:
        import wandb
        if wandb.run is None:
            return
        if not _wandb_ready:
            wandb.define_metric("fork/step")
            wandb.define_metric("fork/*", step_metric="fork/step")
            _wandb_ready = True
        payload = {"fork/step": line["step"], "fork/mean_reward": line["mean_reward"],
                   "fork/mean_recall": line["mean_recall"], "fork/mean_precision": line["mean_precision"]}
        if line.get("precision") == "credit":
            payload.update({f"fork/{k}": line[k] for k in _WANDB_CREDIT_KEYS if line.get(k) is not None})
        wandb.log({k: v for k, v in payload.items() if v is not None}, commit=False)
    except Exception as e:  # noqa: BLE001 — telemetry is never fatal
        print(f"[fork-reward] WARN: wandb log skipped ({e})")


def compute_score(data_sources, solution_strs, ground_truths, extra_infos, **kwargs) -> list[float]:
    """verl batch-reward entry point (reward_model.reward_manager=batch,
    custom_reward_function.path=<this file>, .name=compute_score). Takes
    parallel lists over the batch, returns list[float] in input order."""
    reward = _require_env("FORK_REWARD")
    if reward not in REWARDS:
        raise ValueError(f"FORK_REWARD must be {'|'.join(REWARDS)}, got {reward!r}")
    beta: float | None = None
    precision: str | None = None
    if reward == "fbeta":
        beta = _beta()
        precision = _require_env("FORK_PRECISION")
        if precision not in PRECISIONS:
            raise ValueError(f"FORK_PRECISION must be {'|'.join(PRECISIONS)}, got {precision!r}")
    else:
        for var in ("FORK_BETA", "FORK_PRECISION"):
            if os.environ.get(var, "").strip():
                raise ValueError(f"{var}={os.environ[var].strip()!r} is set but FORK_REWARD=recall — "
                                 f"beta and precision apply to reward: fbeta only")
    domain = _domain()
    if precision == "generic" and domain != "medical":
        raise ValueError("FORK_PRECISION=generic is medical-only (the Generic Criteria prompt has no "
                         f"{domain} twin)")
    grader_cfg = os.environ.get("FORK_GRADER_CONFIG", "configs/grader/qwen36_35b.yaml")
    urls = [u.strip() for u in _require_env("FORK_GRADER_URLS").split(",") if u.strip()]
    run_dir = Path(_require_env("FORK_RUN_DIR"))
    abort_frac = float(os.environ.get("FORK_ABORT_FRAC", "0.05"))

    abort_marker = run_dir / "ABORT"
    if abort_marker.exists():
        raise RuntimeError(
            f"{abort_marker} exists — a previous batch in this run hit the grader-failure hard "
            f"abort, and the abort is sticky (verl retries the reward fn once after an exception; "
            f"the retry must die instantly, not re-pay the grader calls). Fix the grader, then "
            f"relaunch via scripts/train.sh, which clears the marker.")

    if extra_infos is None:
        extra_infos = [None] * len(solution_strs)
    items = [_parse_item(gt, ei, sol) for gt, ei, sol in zip(ground_truths, extra_infos, solution_strs)]
    step = _next_step(run_dir)
    gcfg = _grader_config(grader_cfg)
    scored = asyncio.run(_score_batch(items, reward, precision, urls, gcfg))  # fresh loop + client
    rewards = _assemble(items, scored, reward, beta)
    line = _write_step_logs(run_dir, step, reward, beta, precision, domain, items, scored, rewards)

    flagged = sum(1 for r in rewards if r.recall_failed or r.precision_failed)
    frac = flagged / len(rewards) if rewards else 0.0
    print(f"[fork-reward step {step}] reward={reward} n={len(rewards)} "
          f"mean_reward={line['mean_reward']} mean_recall={line['mean_recall']} "
          f"mean_precision={line['mean_precision']} flagged={flagged} ({frac:.1%})")
    if frac > abort_frac:
        abort_marker.write_text(json.dumps(
            {"step": step, "flagged": flagged, "n": len(rewards),
             "frac": round(frac, 4), "time": round(time.time(), 3)}) + "\n")
        raise RuntimeError(
            f"{flagged}/{len(rewards)} rollouts ({frac:.1%}) hit the grader-failure fallback "
            f"(> {abort_frac:.0%}): a grader outage or systematic breakage — aborting instead of "
            f"training on degenerate rewards. See {run_dir}/{IO_DIR}/step_{step:06d}.jsonl")
    return [float(r.reward) for r in rewards]
