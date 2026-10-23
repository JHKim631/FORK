"""The out-of-domain evaluation (paper Section 4.1, Tables 1, 2 and 4): each
benchmark graded the benchmark's own way — its prompt, its call pattern, its
score — on the responses fork.eval.generate produced on the pool (--pool).

HealthBench (prompts/eval/healthbench_grader.txt = simple-evals' grader
template): ONE call per rubric item, a single user message holding the
conversation plus the graded response rendered "role: content" and the item
as "[points] criterion"; the answer is a JSON object with a boolean
criteria_met (up to 3 attempts, then the item — and the example — is failed,
never guessed). Example score = Σ points of the met items / Σ positive points;
the benchmark's headline is the mean over examples clipped to [0, 1].
HealthBench's own GPT-5.4 grader settings (a pinned snapshot, reasoning
effort low) come from the eval config's `ood.grader_overrides` and apply to
the gpt54 judge here only.

ResearchQA (prompts/eval/researchqa_coverage.txt = compute_coverage.py's
prompt): a single user message per batch of <= 8 rubric items (unnumbered,
one per line) with the response; the judge answers one of "Not at all,
Barely, Moderately, Mostly, Completely" per line (1..5, normalised
(x - 1) / 4); a batch whose line count is off is retried (3 tries) and the
query is skipped if it still is, as in the official code. Query coverage = the
mean over its items; the headline = the mean over queries.

Outputs, in --out-dir: graded.jsonl (one record per example, with every
item's verdict / phrase and every raw judge body), summary.json, manifest.json.
A >= 98 % example gate keeps a bad run uncommitted; ok answers are cached
beside the target, so a rerun only asks for the rest.

    python -m fork.eval.ood --config configs/eval/medical.yaml \\
        --responses outputs/eval/<label>/ood_responses.jsonl --label <label> \\
        --out-dir outputs/eval/<label>/ood.gpt54 [--judge local --base-urls ...]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from ..data.ood import render_conversation
from ..grader.parsing import loads_lenient
from ..grader.templates import PROMPTS_DIR
from ._chunks import PARTIAL, CallCache, install_stop, run_chunk
from ._io import commit_jsonl, merge_manifest, read_jsonl
from .judge_client import Judge, add_judge_args, judge_ident, resolve_judge

__all__ = ["BENCH_TEMPLATES", "RQ_PHRASES", "RQ_BATCH", "raw_user_section", "hb_messages", "hb_parse",
           "hb_score", "rq_messages", "rq_parse", "rq_coverage", "example_units", "assemble_example", "summarize"]

BENCH_TEMPLATES = {"healthbench": "eval/healthbench_grader", "researchqa": "eval/researchqa_coverage"}
RQ_PHRASES = {"Not at all": 1, "Barely": 2, "Moderately": 3, "Mostly": 4, "Completely": 5}
RQ_BATCH = 8          # compute_coverage.py --batch_size default
ATTEMPTS = 3          # per item (HealthBench) / per batch (ResearchQA; the official @retry(tries=3))
_RQ_FIELDS = re.compile(r"\{(response|questions)\}")


def raw_user_section(name: str) -> str:
    """The text after the ===USER=== line of prompts/<name>.txt, the header
    stripped, NOT stripped of whitespace — the official constants carry their
    own leading / trailing newlines and are sent as-is."""
    lines = (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("#"):
        i += 1
    body = "".join(lines[i:])
    marker = "===USER===\n"
    if marker not in body:
        raise ValueError(f"template {name} has no {marker.strip()} line")
    return body[body.index(marker) + len(marker):]


# ---------------------------------------------------------------- HealthBench

def _points_str(w: float) -> str:
    """simple-evals renders the JSON number as-is: the file's ints stay ints."""
    return str(int(w)) if float(w).is_integer() else str(w)


def hb_messages(template: str, messages: list[dict], response: str, item: dict) -> list[dict]:
    """The official grader call: one user message, <<conversation>> = the turns
    plus the response as the assistant turn, <<rubric_item>> = "[points] criterion"."""
    convo = render_conversation(list(messages) + [{"role": "assistant", "content": response}])
    text = template.replace("<<conversation>>", convo).replace(
        "<<rubric_item>>", f"[{_points_str(item['weight'])}] {item['text']}")
    return [{"role": "user", "content": text}]


def hb_parse(raw: str) -> bool | None:
    """A JSON object with a boolean criteria_met -> that bool; anything else None."""
    try:
        obj = loads_lenient(raw)
    except Exception:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("criteria_met"), bool):
        return obj["criteria_met"]
    return None


def hb_score(items: list[tuple[float, bool]]) -> float | None:
    """simple-evals calculate_score: Σ points of the met items / Σ positive points."""
    total = sum(p for p, _ in items if p > 0)
    if total == 0:
        return None
    return sum(p for p, met in items if met) / total


# ----------------------------------------------------------------- ResearchQA

def rq_messages(template: str, response: str, questions: list[str]) -> list[dict]:
    """compute_coverage.py's build_prompt: one user message, the items unnumbered one per line."""
    fill = {"response": response, "questions": "\n".join(questions)}
    return [{"role": "user", "content": _RQ_FIELDS.sub(lambda m: fill[m.group(1)], template)}]


def rq_parse(raw: str, n: int) -> list[int] | None:
    """One phrase per non-empty line, exactly n of them, each one of the five -> 1..5; else None."""
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    if len(lines) != n:
        return None
    scores = [RQ_PHRASES.get(line) for line in lines]
    return None if any(s is None for s in scores) else scores


def rq_coverage(scores: list[int]) -> float | None:
    """Mean of (x - 1) / 4 over a query's items."""
    return sum((s - 1) / 4 for s in scores) / len(scores) if scores else None


# ------------------------------------------------------------------ grading

async def _ask(call, sem, messages: list[dict], parse):
    """Up to ATTEMPTS calls until `parse` accepts the body; (value, raws, error)."""
    raws: list[str] = []
    for _ in range(ATTEMPTS):
        async with sem:
            try:
                raw, _finish = await call(messages)
            except Exception as e:  # noqa: BLE001 — recorded, the caller counts it
                return None, raws, f"{type(e).__name__}: {e}"
        raws.append(raw)
        value = parse(raw)
        if value is not None:
            return value, raws, None
    return None, raws, f"no valid judgment in {ATTEMPTS} attempts"


def example_units(bench: str, template: str, row: dict) -> list[tuple[int, list[dict], object]]:
    """One pool row's judge calls in canonical order, as (idx, messages, parse) —
    HealthBench one per rubric item (idx = its position), ResearchQA one per
    batch of <= RQ_BATCH items (idx = the batch's first item)."""
    if bench == "healthbench":
        turns = json.loads(row["messages"])
        return [(i, hb_messages(template, turns, row["response"], item), hb_parse)
                for i, item in enumerate(row["rubrics"])]
    if bench == "researchqa":
        items = list(row["rubrics"])
        return [(lo, rq_messages(template, row["response"], [it["text"] for it in items[lo:lo + RQ_BATCH]]),
                 lambda raw, n=len(items[lo:lo + RQ_BATCH]): rq_parse(raw, n))
                for lo in range(0, len(items), RQ_BATCH)]
    raise ValueError(f"unknown bench {bench!r}")


def assemble_example(bench: str, row: dict, answers: dict[int, dict]) -> dict:
    """One pool row + its calls' answers ({idx: {value, raws, error}}, value None
    = no valid judgment) -> the example record."""
    rec = {"prompt_id": row["prompt_id"], "label": row["label"], "n_items": len(row["rubrics"]),
           "response_chars": len(row["response"]), "ok": False, "error": None, "score": None, "items": []}
    if bench == "healthbench":
        for i, item in enumerate(row["rubrics"]):
            a = answers[i]
            rec["items"].append({"text": item["text"], "points": item["weight"], "tags": list(item.get("tags") or []),
                                 "met": a["value"], "raws": a["raws"], "error": a["error"]})
        if all(it["met"] is not None for it in rec["items"]):
            rec["score"] = hb_score([(it["points"], it["met"]) for it in rec["items"]])
            rec["ok"] = rec["score"] is not None
            if not rec["ok"]:
                rec["error"] = "no positive points"
        else:
            rec["error"] = f"{sum(it['met'] is None for it in rec['items'])} item(s) without a verdict"
        return rec
    if bench == "researchqa":
        items = list(row["rubrics"])
        rec["batches"] = []
        scores: list[int] = []
        for lo in range(0, len(items), RQ_BATCH):
            a = answers[lo]
            rec["batches"].append({"lo": lo, "n": len(items[lo:lo + RQ_BATCH]), "raws": a["raws"], "error": a["error"]})
            if a["value"] is None:
                rec["error"] = f"batch at {lo}: {a['error']}"
                break                                   # the official script skips the query here
            scores.extend(a["value"])
        for it, s in zip(items, scores):
            rec["items"].append({"text": it["text"], "types": list(it.get("types") or []), "score": s})
        if len(scores) == len(items):
            rec["score"] = rq_coverage(scores)
            kept = [s for it, s in zip(items, scores) if "Citation" not in (it.get("types") or [])]
            rec["coverage_no_citation"] = rq_coverage(kept)
            rec["ok"] = True
        return rec
    raise ValueError(f"unknown bench {bench!r}")


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summarize(bench: str, label: str, judge_model: str, records: list[dict]) -> dict:
    n_ok = sum(1 for r in records if r["ok"])
    out = {"bench": bench, "label": label, "judge_model": judge_model, "n": len(records), "n_ok": n_ok,
           "parse_success": round(n_ok / len(records), 4) if records else None,
           "n_items": sum(r["n_items"] for r in records),
           "mean_score": _mean([r["score"] for r in records if r["ok"]]),
           "mean_response_chars": _mean([float(r["response_chars"]) for r in records])}
    if bench == "healthbench":
        m = out["mean_score"]
        out["mean_score_clipped"] = None if m is None else round(min(1.0, max(0.0, m)), 4)
        out["n_calls"] = sum(len(it["raws"]) for r in records for it in r["items"])
    else:
        out["mean_coverage_no_citation"] = _mean([r.get("coverage_no_citation") for r in records if r["ok"]])
        out["n_calls"] = sum(len(b["raws"]) for r in records for b in r.get("batches", []))
    return out


def load_rows(responses: str, pool: str, label: str) -> list[dict]:
    """The pool rows joined with a generation's responses (fork.eval.generate --pool)."""
    by_id = {r["prompt_id"]: r for r in pq.read_table(pool).to_pylist()}
    rows = []
    for rec in read_jsonl(responses):
        if rec["prompt_id"] not in by_id:
            raise SystemExit(f"{rec['prompt_id']} not in {pool} — wrong pool?")
        row = dict(by_id[rec["prompt_id"]])
        row.update(response=rec["response"], label=label)
        rows.append(row)
    return rows


async def main_async(args) -> int:
    eval_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ood = eval_cfg.get("ood") or {}
    bench, pool = ood.get("bench"), ood.get("pool")
    if bench not in BENCH_TEMPLATES or not pool:
        raise SystemExit(f"{args.config}: `ood` must name a bench ({' | '.join(BENCH_TEMPLATES)}) and its pool")
    judge_cfg = resolve_judge(eval_cfg, args.judge, args.base_urls, args.concurrency, bench=bench)
    judge = Judge(judge_cfg)  # first: fail fast on a missing key / replica URLs
    template_name = BENCH_TEMPLATES[bench]
    template = raw_user_section(template_name)
    rows = load_rows(args.responses, pool, args.label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    units = [example_units(bench, template, row) for row in rows]
    calls = [(row["prompt_id"], idx, msgs, parse) for row, us in zip(rows, units) for idx, msgs, parse in us]
    effort = judge_cfg.get("reasoning_effort")
    print(f"[ood] {bench} {args.label}: {len(rows)} responses, {sum(len(r['rubrics']) for r in rows)} rubric items, "
          f"{len(calls)} calls via {judge_cfg['model']}" + (f" (reasoning effort {effort})" if effort else ""))

    target = out_dir / "graded.jsonl"
    cache = CallCache(target, ("prompt_id", "idx"))
    cached = cache.load()
    pending = [c for c in calls if c[:2] not in cached]
    if cached:
        print(f"[ood] {len(cached)} calls cached, {len(pending)} pending")
    conc = int(judge_cfg.get("concurrency", 8))
    sem = asyncio.Semaphore(conc)
    stop = install_stop()

    async def ask(c) -> dict:
        pid, idx, msgs, parse = c
        value, raws, err = await _ask(judge.chat, sem, msgs, parse)
        return {"prompt_id": pid, "idx": idx, "ok": value is not None, "value": value, "raws": raws, "error": err}

    try:
        fresh = await run_chunk([(c[:2], c) for c in pending], ask, conc, stop,
                                lambda _k, rec: cache.append(rec) if rec["ok"] else None)
    finally:
        await judge.close()
    answered = {**cached, **fresh}
    n_have = sum(1 for c in calls if c[:2] in answered)
    if n_have < len(calls):
        print(f"[ood] partial: {n_have}/{len(calls)} calls answered" + (" (stopped)" if stop.is_set() else "")
              + f" — cache {cache.path}; rerun to continue")
        return PARTIAL
    records = [assemble_example(bench, row, {idx: answered[(row["prompt_id"], idx)] for idx, _m, _p in us})
               for row, us in zip(rows, units)]
    summary = summarize(bench, args.label, judge_cfg["model"], records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["parse_success"] is not None and summary["parse_success"] < 0.98:
        print("WARNING: >2% examples without a full set of verdicts — inspect errors; graded.jsonl not committed"
              f" (ok answers stay cached in {cache.path}; a rerun retries the failed calls)")
        return 1
    commit_jsonl(target, records)
    merge_manifest(out_dir, {"lane": out_dir.name, "judge": judge_ident(judge_cfg), "template": template_name,
                             "bench": bench, "pool": pool, "eval_cfg": args.config},
                   "runs", args.label, {"n": summary["n"], "n_ok": summary["n_ok"], "n_calls": summary["n_calls"],
                                        "from_cache": len(cached)})
    cache.discard()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade one model's OOD-pool responses the benchmark's own way")
    ap.add_argument("--config", required=True, help="configs/eval/<domain>.yaml (its `ood` block names the bench and pool)")
    ap.add_argument("--responses", required=True, help="<label dir>/ood_responses.jsonl")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True, help="the lane dir, e.g. outputs/eval/<label>/ood.gpt54")
    add_judge_args(ap)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
