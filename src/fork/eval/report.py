"""The results of one model, in the paper's columns (Tables 1 and 2), from the
lane files under outputs/eval/<label>/:

    recall, precision, F1     rubric_based.<judge>/{recall,precision}_scored.jsonl — macro means over
                              the grader subset; F1 is the mean of the per-response F1 of recall
                              and precision (both halves ok)
    win rate                  rubric_free.<judge>.<prompt>/vs_<other>.jsonl — the win rate of the
                              label against the comparator, ties excluded, position-flip averaged
    length                    the mean response length in characters — over trajectory_responses.jsonl
                              (the 300-prompt subset, the paper's length column) when the trajectory
                              was generated, else grader_responses.jsonl (100 prompts); `length_n` says which
    OOD recall                ood.<judge>/summary.json — HealthBench's clipped mean score, or
                              ResearchQA's mean coverage
    OOD win rate              ood_rubric_free.<judge>.<prompt>/vs_<other>.jsonl

plus the per-dimension Likert deltas of a pair (Table 8), the presence / absence
satisfaction rates (Table 3; needs the rubric-types artifact), and the
per-checkpoint trajectory of a run (Figure 1). 95 % CIs are seeded percentile
bootstraps over prompts.

    python -m fork.eval.report --label fork_beta1.25_medical@291 --vs base_medical
    python -m fork.eval.report --label fork_beta1.25_medical@291 --vs base_medical --dimensions
    python -m fork.eval.report --trajectory fork_beta1.25_medical --vs base_medical --judge local
    python -m fork.eval.report --label fork_beta1.25_medical@291 --rubric-types outputs/rubric_types/medical.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from ._io import read_jsonl
from .rubric_free import DIMENSIONS

__all__ = ["bootstrap_ci", "per_prompt_deltas", "pair_report", "per_prompt_verdicts", "verdict_pair_report",
           "judgment_format", "rubric_based_row", "run_row", "render_rows", "trajectory_rows"]


# ------------------------------------------------------------------ pairwise

def dimensions_of(records: list[dict]) -> tuple[str, ...]:
    for rec in records:
        if rec.get("ok") and isinstance(rec.get("x"), dict):
            return tuple(rec["x"])
    return DIMENSIONS


def per_prompt_deltas(records: list[dict]) -> list[dict]:
    """Position-averaged per-prompt rows: {prompt_id, delta: {dim: float},
    winner: "x"|"y"|"tie"}. Prompts with no ok judgment are dropped; a single
    ok order still counts (averaged over what exists)."""
    dims = dimensions_of(records)
    by_pid: dict[str, list[dict]] = {}
    for rec in records:
        if rec.get("ok"):
            by_pid.setdefault(rec["prompt_id"], []).append(rec)
    rows = []
    for pid, recs in sorted(by_pid.items()):
        avg = {side: {dim: sum(r[side][dim] for r in recs) / len(recs) for dim in dims} for side in ("x", "y")}
        delta = {dim: avg["x"][dim] - avg["y"][dim] for dim in dims}
        overall = delta["overall"]
        winner = "tie" if overall == 0 else ("x" if overall > 0 else "y")
        rows.append({"prompt_id": pid, "n_orders": len(recs), "delta": delta, "winner": winner,
                     "x": avg["x"], "y": avg["y"]})
    return rows


def bootstrap_ci(values: list[float], n_boot: int = 10_000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Seeded percentile bootstrap CI for the mean, over prompts."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return (lo, hi)


def pair_report(records: list[dict], x_label: str, y_label: str, n_boot: int = 10_000, seed: int = 0) -> dict:
    """The Likert pair: per-dimension deltas (x − y) with CIs, the mean absolute
    scores of each side, and the win rate of x (ties excluded)."""
    rows = per_prompt_deltas(records)
    dims = {}
    for dim in dimensions_of(records):
        vals = [r["delta"][dim] for r in rows]
        mean = sum(vals) / len(vals) if vals else float("nan")
        lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        dims[dim] = {"delta": round(mean, 3), "ci95": [round(lo, 3), round(hi, 3)],
                     "x_mean": round(sum(r["x"][dim] for r in rows) / len(rows), 3) if rows else None,
                     "y_mean": round(sum(r["y"][dim] for r in rows) / len(rows), 3) if rows else None}
    wins = sum(1 for r in rows if r["winner"] == "x")
    losses = sum(1 for r in rows if r["winner"] == "y")
    ties = sum(1 for r in rows if r["winner"] == "tie")
    decided = wins + losses
    lo, hi = bootstrap_ci([1.0] * wins + [0.0] * losses, n_boot=n_boot, seed=seed)
    return {"pair": f"{x_label} vs {y_label}", "x": x_label, "y": y_label, "format": "likert",
            "n_prompts": len(rows), "dimensions": dims,
            "win_rate": {"rate": round(wins / decided, 4) if decided else None,
                         "ci95": [round(lo, 4), round(hi, 4)] if decided else None,
                         "wins": wins, "losses": losses, "ties": ties}}


def judgment_format(records: list[dict]) -> str:
    return "verdict" if records and "verdict" in records[0] else "likert"


def per_prompt_verdicts(records: list[dict]) -> list[dict]:
    """The position-flip rule for verdict judgments (MT-Bench's pairwise
    summary): per prompt, both orders say x -> winner x, both say y -> y,
    anything else (a tie in either order, a disagreement, one order missing
    or failed) -> tie."""
    by_pid: dict[str, dict[str, str]] = {}
    for rec in records:
        if rec.get("ok"):
            by_pid.setdefault(rec["prompt_id"], {})[rec["order"]] = rec["verdict"]
    rows = []
    for pid, verdicts in sorted(by_pid.items()):
        agreed = set(verdicts.values())
        winner = agreed.pop() if len(verdicts) == 2 and len(agreed) == 1 and agreed != {"tie"} else "tie"
        rows.append({"prompt_id": pid, "n_orders": len(verdicts), "verdicts": verdicts, "winner": winner})
    return rows


def verdict_pair_report(records: list[dict], x_label: str, y_label: str, n_boot: int = 10_000, seed: int = 0) -> dict:
    rows = per_prompt_verdicts(records)
    wins = sum(1 for r in rows if r["winner"] == "x")
    losses = sum(1 for r in rows if r["winner"] == "y")
    ties = sum(1 for r in rows if r["winner"] == "tie")
    decided = wins + losses
    lo, hi = bootstrap_ci([1.0] * wins + [0.0] * losses, n_boot=n_boot, seed=seed)
    per_order = {o: {v: sum(1 for r in rows if r["verdicts"].get(o) == v) for v in ("x", "y", "tie")}
                 for o in ("xy", "yx")}
    return {"pair": f"{x_label} vs {y_label}", "x": x_label, "y": y_label, "format": "verdict",
            "n_prompts": len(rows),
            "win_rate": {"rate": round(wins / decided, 4) if decided else None,
                         "ci95": [round(lo, 4), round(hi, 4)] if decided else None,
                         "wins": wins, "losses": losses, "ties": ties},
            "per_order": per_order}


def any_pair_report(records: list[dict], x_label: str, y_label: str, n_boot: int = 10_000, seed: int = 0) -> dict:
    if judgment_format(records) == "verdict":
        return verdict_pair_report(records, x_label, y_label, n_boot, seed)
    return pair_report(records, x_label, y_label, n_boot, seed)


# ------------------------------------------------------------------ one label

def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def rubric_based_row(lane_dir: Path) -> dict:
    """Macro recall, macro precision and the mean per-response F1 of one label's
    rubric-based lane; None for a half that is not on disk."""
    out = {"recall": None, "precision": None, "f1": None, "n_recall": 0, "n_precision": 0, "n_f1": 0}
    rec_path, pre_path = lane_dir / "recall_scored.jsonl", lane_dir / "precision_scored.jsonl"
    recall = {r["prompt_id"]: r["recall"] for r in read_jsonl(rec_path) if r.get("ok")} if rec_path.exists() else {}
    precision = {r["prompt_id"]: r["precision"] for r in read_jsonl(pre_path) if r.get("ok")} if pre_path.exists() else {}
    if recall:
        out.update(recall=_mean(recall.values()), n_recall=len(recall))
    if precision:
        out.update(precision=_mean(precision.values()), n_precision=len(precision))
    both = [pid for pid in recall if pid in precision]
    if both:
        f1s = [2 * recall[p] * precision[p] / (recall[p] + precision[p]) if recall[p] + precision[p] > 0 else 0.0
               for p in both]
        out.update(f1=_mean(f1s), n_f1=len(both))
    return out


def _lane_records(path: Path) -> list[dict] | None:
    return read_jsonl(path) if path.exists() else None


def _length(d: Path, files: tuple[str, ...]) -> tuple[float | None, int]:
    """(mean response length in characters, the number of responses) from the
    first of `files` that exists under the label dir; (None, 0) when none does."""
    for name in files:
        if (d / name).exists():
            recs = read_jsonl(d / name)
            return _mean([len(r["response"]) for r in recs]), len(recs)
    return None, 0


def run_row(eval_dir: Path, label: str, vs: str | None, judge: str = "gpt54", prompt: str = "p1") -> dict:
    """One label's row in the paper's columns, from whatever lanes are on disk."""
    d = eval_dir / label
    row = {"label": label, "vs": vs, "judge": judge, "prompt": prompt}
    row.update(rubric_based_row(d / f"rubric_based.{judge}"))
    row["length"], row["length_n"] = _length(d, ("trajectory_responses.jsonl", "grader_responses.jsonl"))
    row["win_rate"] = row["win_rate_detail"] = None
    if vs:
        recs = _lane_records(d / f"rubric_free.{judge}.{prompt}" / f"vs_{vs}.jsonl")
        if recs:
            rep = any_pair_report(recs, label, vs)
            row["win_rate"], row["win_rate_detail"] = rep["win_rate"]["rate"], rep["win_rate"]
    summ = d / f"ood.{judge}" / "summary.json"
    row["ood_recall"] = row["ood_bench"] = None
    if summ.exists():
        s = json.loads(summ.read_text())
        row["ood_bench"] = s["bench"]
        row["ood_recall"] = s["mean_score_clipped"] if s["bench"] == "healthbench" else s["mean_score"]
    row["ood_win_rate"] = None
    if vs:
        recs = _lane_records(d / f"ood_rubric_free.{judge}.{prompt}" / f"vs_{vs}.jsonl")
        if recs:
            row["ood_win_rate"] = any_pair_report(recs, label, vs)["win_rate"]["rate"]
    return row


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}"


def _chars(x) -> str:
    return "—" if x is None else f"{x:,.0f}"


def render_rows(rows: list[dict]) -> str:
    vs = next((r["vs"] for r in rows if r.get("vs")), None)
    bench = next((r["ood_bench"] for r in rows if r.get("ood_bench")), "OOD")
    head = (f"| model | recall | precision | F1 | win rate vs {vs or '—'} | length (chars) | "
            f"{bench} recall | {bench} win rate |")
    lines = [head, "|---" * 8 + "|"]
    for r in rows:
        lines.append(f"| {r['label']} | {_pct(r['recall'])} | {_pct(r['precision'])} | {_pct(r['f1'])} | "
                     f"{_pct(r['win_rate'])} | {_chars(r['length'])} | "
                     f"{_pct(r['ood_recall'])} | {_pct(r['ood_win_rate'])} |")
    return "\n".join(lines) + "\n"


def render_dimensions(rep: dict) -> str:
    dims = list(rep["dimensions"])
    lines = [f"| {rep['pair']} | " + " | ".join(dims) + " |", "|---" * (len(dims) + 1) + "|"]
    lines.append(f"| {rep['x']} (mean) | " + " | ".join(f"{rep['dimensions'][d]['x_mean']:.2f}" for d in dims) + " |")
    lines.append(f"| {rep['y']} (mean) | " + " | ".join(f"{rep['dimensions'][d]['y_mean']:.2f}" for d in dims) + " |")
    lines.append("| delta [95% CI] | " + " | ".join(
        f"{rep['dimensions'][d]['delta']:+.2f} [{rep['dimensions'][d]['ci95'][0]:+.2f}, {rep['dimensions'][d]['ci95'][1]:+.2f}]"
        for d in dims) + " |")
    w = rep["win_rate"]
    lines.append(f"\nwin rate of {rep['x']}: {_pct(w['rate'])} % ({w['wins']}W / {w['losses']}L / {w['ties']}T), "
                 f"95% CI [{_pct(w['ci95'][0])}, {_pct(w['ci95'][1])}] over {rep['n_prompts']} prompts")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- trajectory

_STEP_RE = re.compile(r"@(\d+)$")


def trajectory_rows(eval_dir: Path, run: str, vs: str | None, judge: str = "local", prompt: str = "p1") -> list[dict]:
    """One row per checkpoint of a run (every <run>@<step> dir under eval_dir),
    from its rubric_based.<judge> lane on the trajectory subset and its
    pairwise lane against `vs`."""
    rows = []
    for d in sorted(eval_dir.glob(f"{run}@*")):
        m = _STEP_RE.search(d.name)
        if not m:
            continue
        row = {"step": int(m.group(1)), "label": d.name}
        row.update(rubric_based_row(d / f"rubric_based.{judge}"))
        row["length"], row["length_n"] = _length(d, ("trajectory_responses.jsonl",))
        row["win_rate"] = None
        if vs:
            recs = _lane_records(d / f"rubric_free.{judge}.{prompt}" / f"vs_{vs}.jsonl")
            if recs:
                row["win_rate"] = any_pair_report(recs, d.name, vs)["win_rate"]["rate"]
        rows.append(row)
    return sorted(rows, key=lambda r: r["step"])


def render_trajectory(rows: list[dict], vs: str | None) -> str:
    lines = [f"| step | recall | precision | F1 | win rate vs {vs or '—'} | length (chars) |", "|---" * 6 + "|"]
    for r in rows:
        lines.append(f"| {r['step']} | {_pct(r['recall'])} | {_pct(r['precision'])} | {_pct(r['f1'])} | "
                     f"{_pct(r['win_rate'])} | {_chars(r['length'])} |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Results in the paper's columns, from the evaluation lanes on disk")
    ap.add_argument("--eval-dir", default="outputs/eval")
    ap.add_argument("--label", action="append", default=[], help="a label (repeatable): <run>@<step> or base_<domain>")
    ap.add_argument("--trajectory", default=None, metavar="RUN", help="every checkpoint of a run instead of labels")
    ap.add_argument("--vs", default=None, help="the comparator label of the pairwise lanes (e.g. base_medical)")
    ap.add_argument("--judge", default=None, help="the lanes' judge: gpt54 or local (default gpt54; local for --trajectory)")
    ap.add_argument("--prompt", default="p1", help="the pairwise prompt id (p1 default, p2)")
    ap.add_argument("--dimensions", action="store_true", help="also print the per-dimension Likert report of each pair")
    ap.add_argument("--rubric-types", default=None, metavar="ARTIFACT",
                    help="also print the presence / absence satisfaction rates from this typing artifact")
    ap.add_argument("--json", action="store_true", help="print the rows as JSON instead of markdown")
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    judge = args.judge or ("local" if args.trajectory else "gpt54")
    if args.trajectory:
        rows = trajectory_rows(eval_dir, args.trajectory, args.vs, judge, args.prompt)
        print(json.dumps(rows, indent=2) if args.json else render_trajectory(rows, args.vs))
        return
    if not args.label:
        raise SystemExit("give --label (repeatable) or --trajectory <run>")
    rows = [run_row(eval_dir, lab, args.vs, judge, args.prompt) for lab in args.label]
    print(json.dumps(rows, indent=2) if args.json else render_rows(rows))
    if args.dimensions and args.vs:
        for lab in args.label:
            path = eval_dir / lab / f"rubric_free.{judge}.{args.prompt}" / f"vs_{args.vs}.jsonl"
            if path.exists():
                recs = read_jsonl(path)
                if judgment_format(recs) == "likert":
                    print(render_dimensions(pair_report(recs, lab, args.vs)))
    if args.rubric_types:
        from .rubric_types import grader_fields, load_types

        types = load_types(args.rubric_types)
        for lab in args.label:
            path = eval_dir / lab / f"rubric_based.{judge}" / "recall_scored.jsonl"
            if path.exists():
                f = grader_fields(read_jsonl(path), types)
                print(f"{lab}: presence-based satisfaction {_pct(f['presence_met_rate'])} %, "
                      f"absence-based {_pct(f['absence_met_rate'])} % "
                      f"({f['n_presence_items']} / {f['n_absence_items']} items over {f['n_typed_responses']} responses)")


if __name__ == "__main__":
    main()
