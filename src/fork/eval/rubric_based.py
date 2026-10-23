"""The rubric-based evaluation (paper Section 4.1): rubric recall and response
precision of one model's responses on the grader subset, under one judge.

Recall: every response is graded against its prompt's rubrics with the SAME
prompt the training reward uses (prompts/train/coverage_grading.txt) and scored
with the weighted recall (fork.rewards.recall). Precision: the response is
unitized (fork.rewards.unitize) and every unit labeled correct / necessary by
the precision labeler (prompts/eval/precision_label.txt — the Generic Criteria
prompt, Appendix E.2 — or its science twin), chunked exactly as the training
labeler is (all units rendered in every call, 64 judged per call, the seeded
sample above 256 units, ANY chunk failing fails the response, a refusal
re-sent once through the audit twin); precision = the fraction of units with
both (fork.rewards.generic). The per-response F1 of recall and precision, and
the macro means, are computed by fork.eval.report.

The judge is the lane's: `gpt54` (the reference grader) or `local` (the
training grader on the served replicas), so recall and precision in one lane
share one grader identity. Outputs, in --out-dir: recall_scored.jsonl,
precision_scored.jsonl (one record per response, with the raw judge body),
summary.json, manifest.json. A >= 98 % parse gate keeps a bad run uncommitted;
ok answers are cached beside the target, so a rerun only asks for the rest.

    python -m fork.eval.rubric_based --config configs/eval/medical.yaml \\
        --responses outputs/eval/<label>/grader_responses.jsonl --label <label> \\
        --out-dir outputs/eval/<label>/rubric_based.gpt54 [--judge local --base-urls ...] [--measure recall|precision]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from ..data.schemas import Rubric
from ..grader.client import is_refusal
from ..grader.parsing import ParseError
from ..grader.templates import (
    PROMPTS_DIR,
    load_template,
    render_reference_block,
    render_rubric_list,
    render_units_block,
)
from ..rewards.generic import CRITERIA, generic_precision, parse_unit_labels_salvage
from ..rewards.precision import LABEL_CHUNK, MAX_UNITS, chunk_ranges, select_label_sample
from ..rewards.recall import recall_from_text
from ..rewards.unitize import split
from ._chunks import PARTIAL, CallCache, install_stop, run_chunk
from ._io import commit_jsonl, merge_manifest, read_jsonl
from .judge_client import Judge, add_judge_args, judge_ident, resolve_judge

__all__ = ["RECALL_TEMPLATE", "PRECISION_TEMPLATES", "AUDIT_TEMPLATES", "load_rows", "grade_recall_one",
           "score_precision_one", "summarize_recall", "summarize_precision"]

RECALL_TEMPLATE = "train/coverage_grading"
PRECISION_TEMPLATES = {"medical": "eval/precision_label", "science": "eval/precision_label_science"}
AUDIT_TEMPLATES = {"medical": "eval/precision_label_audit"}   # the refusal-rescue twin (medical only)


def load_rows(responses_path: str | Path, eval_pool: str | Path) -> list[dict]:
    """Join generated responses with the eval parquet by prompt_id (the rubrics
    live only in the parquet). A missing id is a loud error — the responses
    were not generated from this pool."""
    recs = read_jsonl(responses_path)
    tbl = pq.read_table(eval_pool, columns=["prompt_id", "prompt", "rubrics", "reference_answer"])
    by_id = {r["prompt_id"]: r for r in tbl.to_pylist()}
    missing = [r["prompt_id"] for r in recs if r["prompt_id"] not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)}/{len(recs)} prompt_ids not in {eval_pool} (e.g. {missing[0]}) — wrong eval pool?")
    rows = []
    for rec in recs:
        row = dict(by_id[rec["prompt_id"]])
        row["response"] = rec["response"]
        row["response_tokens"] = rec.get("response_tokens")
        rows.append(row)
    return rows


async def grade_recall_one(call, template, sem, row: dict) -> dict:
    """One response's recall. `call` is an async (messages) -> (raw_text,
    finish_reason) — Judge.chat, or any stand-in with that signature. A body cut off at the
    token cap is refused, never parsed: missing rubric entries would silently
    deflate the score."""
    rubrics = [Rubric(text=x["text"], weight=float(x["weight"])) for x in row["rubrics"]]
    rec = {"prompt_id": row["prompt_id"], "label": row["label"],
           "n_rubrics": len(rubrics), "response_chars": len(row["response"]),
           "ok": False, "error": None, "finish_reason": None, "raw": "",
           "recall": None, "n_met": None, "n_missing": None}
    msgs = template.messages(prompt=row["prompt"], response=row["response"],
                             rubric_list_string=render_rubric_list([r.text for r in rubrics]))
    async with sem:
        try:
            raw, finish = await call(msgs)
            rec["raw"], rec["finish_reason"] = raw, finish
            if finish == "length":
                raise ValueError("truncated at the token cap (finish_reason=length)")
            res = recall_from_text(rubrics, raw)
            rec.update(ok=True, recall=res.score, n_met=res.n_met, n_missing=res.n_missing)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
    return rec


async def score_precision_one(call, template, sem, row: dict, audit=None) -> dict:
    """One response's precision. `sem` bounds the calls in flight across ALL
    responses, since one response issues a call per chunk. `audit` = the
    refusal-rescue twin (None: no rescue)."""
    units = split(row["response"])
    rec = {"prompt_id": row["prompt_id"], "label": row["label"],
           "n_units": len(units), "n_labeled": 0,
           "response_chars": len(row["response"]), "response_tokens": row.get("response_tokens"),
           "ok": False, "error": None, "finish_reason": None, "raw": "",
           "precision": None, "n_valid": None, "n_missing": None, "refusal_retries": 0}
    if not units:
        # An empty response has precision 0 with no call — a measurement, not a failure.
        res = generic_precision([], {})
        rec.update(ok=True, precision=res.score, n_valid=res.n_valid, n_missing=res.n_missing)
        return rec
    sample = select_label_sample(units, cap=MAX_UNITS)
    rec["n_labeled"] = len(sample)
    rubrics = [Rubric(text=x["text"], weight=float(x["weight"])) for x in row["rubrics"]]
    fields = dict(question=row["prompt"],
                  reference_block=render_reference_block(row.get("reference_answer", "")),
                  rubric_block=render_rubric_list([r.text for r in rubrics]),
                  units_block=render_units_block(sample))

    async def ask(tpl, lo: int, hi: int) -> tuple[str, str | None]:
        async with sem:
            raw, finish = await call(tpl.messages(**fields, judge_range=f"{lo}-{hi}"))
        if finish == "length":
            raise ValueError(f"chunk {lo}-{hi} truncated at the token cap (finish_reason=length)")
        return raw, finish

    async def one(lo: int, hi: int) -> tuple[str, str | None, dict, int]:
        raw, finish = await ask(template, lo, hi)
        try:
            return raw, finish, parse_unit_labels_salvage(raw, hi - lo + 1, first=lo, keys=CRITERIA), 0
        except ParseError:
            if audit is None or not is_refusal(raw):
                raise
        raw, finish = await ask(audit, lo, hi)      # the refusal rescue: the same payload, once
        return raw, finish, parse_unit_labels_salvage(raw, hi - lo + 1, first=lo, keys=CRITERIA), 1

    try:
        chunks = await asyncio.gather(*(one(lo, hi) for lo, hi in chunk_ranges(len(sample), LABEL_CHUNK)))
        labels: dict[int, dict] = {}
        for _raw, _finish, part, _rescued in chunks:
            labels.update(part)
        rec["raw"] = json.dumps({str(i): labels[i] for i in sorted(labels)}, ensure_ascii=False)
        rec["finish_reason"] = chunks[-1][1]
        res = generic_precision(sample, labels, total_units=len(units))
        rec.update(ok=True, precision=res.score, n_valid=res.n_valid, n_missing=res.n_missing,
                   refusal_retries=sum(c[3] for c in chunks))
    except Exception as e:  # ANY chunk failing fails the response — no partial credit
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summarize_recall(label: str, judge_model: str, records: list[dict]) -> dict:
    n_ok = sum(1 for r in records if r["ok"])
    return {"label": label, "judge_model": judge_model, "n": len(records), "n_ok": n_ok,
            "parse_success": round(n_ok / len(records), 4) if records else None,
            "truncated": sum(1 for r in records if r.get("finish_reason") == "length"),
            "mean_recall": _mean([r["recall"] for r in records if r["ok"]]),
            "mean_response_chars": _mean([float(r["response_chars"]) for r in records])}


def summarize_precision(label: str, judge_model: str, records: list[dict]) -> dict:
    n_ok = sum(1 for r in records if r["ok"])
    return {"label": label, "judge_model": judge_model, "criteria": "correct AND necessary",
            "n": len(records), "n_ok": n_ok,
            "parse_success": round(n_ok / len(records), 4) if records else None,
            "truncated": sum(1 for r in records if r.get("finish_reason") == "length"),
            "refusal_retries": sum(r.get("refusal_retries") or 0 for r in records),
            "mean_precision": _mean([r["precision"] for r in records if r["ok"]]),
            "mean_units": _mean([float(r["n_units"]) for r in records]),
            "mean_response_chars": _mean([float(r["response_chars"]) for r in records]),
            "mean_response_tokens": _mean([r["response_tokens"] for r in records])}


async def _run_measure(measure: str, rows: list[dict], judge: Judge, judge_cfg: dict, domain: str,
                       out_dir: Path, label: str, config: str) -> int:
    if measure == "recall":
        template, audit, name = load_template(RECALL_TEMPLATE), None, RECALL_TEMPLATE
        target = out_dir / "recall_scored.jsonl"
    else:
        name = PRECISION_TEMPLATES.get(domain)
        if name is None:
            raise SystemExit(f"no precision labeler for domain {domain!r}")
        template = load_template(name)
        audit_name = AUDIT_TEMPLATES.get(domain)
        audit = load_template(audit_name) if audit_name else None
        target = out_dir / "precision_scored.jsonl"
    print(f"[rubric_based:{measure}] {label}: {len(rows)} responses via {judge_cfg['model']} ({name})")
    cache = CallCache(target, ("prompt_id",))
    cached = cache.load()
    pending = [r for r in rows if cache.key(r) not in cached]
    if cached:
        print(f"[rubric_based:{measure}] {len(cached)} answers cached, {len(pending)} pending")
    conc = int(judge_cfg.get("concurrency", 8))
    sem = asyncio.Semaphore(conc)
    stop = install_stop()
    if measure == "recall":
        make = lambda r: grade_recall_one(judge.chat, template, sem, r)  # noqa: E731
    else:
        make = lambda r: score_precision_one(judge.chat, template, sem, r, audit)  # noqa: E731
    fresh = await run_chunk([(cache.key(r), r) for r in pending], make, conc, stop,
                            lambda _k, rec: cache.append(rec) if rec["ok"] else None)
    answered = {**cached, **fresh}
    records = [answered.get(cache.key(r)) for r in rows]      # the canonical order: the responses file's
    if any(rec is None for rec in records):
        n_have = sum(rec is not None for rec in records)
        print(f"[rubric_based:{measure}] partial: {n_have}/{len(rows)} answered"
              + (" (stopped)" if stop.is_set() else "") + f" — cache {cache.path}; rerun to continue")
        return PARTIAL
    summary = (summarize_recall if measure == "recall" else summarize_precision)(label, judge_cfg["model"], records)
    print(json.dumps(summary, indent=2))
    if summary["parse_success"] is not None and summary["parse_success"] < 0.98:
        print(f"WARNING: >2% {measure} failures — inspect errors before comparing models "
              f"(ok answers stay cached in {cache.path}; a rerun retries the failed calls)")
        return 1
    commit_jsonl(target, records)  # atomic, only after the gate passes
    sha = hashlib.sha256((PROMPTS_DIR / f"{name}.txt").read_bytes()).hexdigest()
    merge_manifest(out_dir, {"lane": out_dir.name, "judge": judge_ident(judge_cfg), "eval_cfg": config,
                             f"{measure}_template": name, f"{measure}_template_sha256": sha},
                   f"{measure}_runs", label, {"n": summary["n"], "n_ok": summary["n_ok"], "from_cache": len(cached)})
    cache.discard()
    # the lane summary: whichever halves exist
    summ_path = out_dir / "summary.json"
    summ = json.loads(summ_path.read_text()) if summ_path.exists() else {"label": label}
    summ[measure] = summary
    summ_path.write_text(json.dumps(summ, indent=2))
    return 0


async def main_async(args) -> int:
    eval_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    judge_cfg = resolve_judge(eval_cfg, args.judge, args.base_urls, args.concurrency)
    judge = Judge(judge_cfg)  # first: fail fast on a missing key / replica URLs
    rows = load_rows(args.responses, eval_cfg["eval_pool"])
    for row in rows:
        row["label"] = args.label
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    measures = ("recall", "precision") if args.measure == "both" else (args.measure,)
    status = 0
    try:
        for measure in measures:
            target = out_dir / f"{measure}_scored.jsonl"
            if target.exists():
                print(f"[rubric_based:{measure}] {target} exists — skipped")
                continue
            status = max(status, await _run_measure(measure, rows, judge, judge_cfg, eval_cfg["domain"],
                                                    out_dir, args.label, args.config))
    finally:
        await judge.close()
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description="Rubric-based evaluation (recall + precision) of one model's responses")
    ap.add_argument("--config", required=True, help="configs/eval/<domain>.yaml")
    ap.add_argument("--responses", required=True, help="fork.eval.generate output (grader_responses.jsonl)")
    ap.add_argument("--label", required=True, help="the model label, e.g. fork_beta1.25_medical@291")
    ap.add_argument("--out-dir", required=True, help="the lane dir, e.g. outputs/eval/<label>/rubric_based.gpt54")
    ap.add_argument("--measure", default="both", choices=("recall", "precision", "both"))
    add_judge_args(ap)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
