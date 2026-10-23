"""The presence-based / absence-based rubric satisfaction rates (paper
Section 4.3, Table 3).

ONE response-independent pass labels every rubric item of a domain's
evaluation subsets (the 300-prompt trajectory subset ∪ the 100-prompt grader
subset) presence-based or absence-based from the item's wording alone
(prompts/eval/rubric_types.txt — no question, no response shown), with the
training grader. The artifact — one record per prompt with the item types in
rubric order — is written to outputs/rubric_types/<domain>.jsonl. The
satisfaction rates are then pure aggregation over the recall records a
rubric-based lane already holds: the per-item verdicts in recall_scored.jsonl's
raw judge bodies, pooled by type over the responses of one label
(`grader_fields`; fork.eval.report prints them).

    python -m fork.eval.rubric_types --config configs/eval/medical.yaml --base-urls http://localhost:8001/v1,...
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from ..grader.parsing import ParseError, loads_lenient
from ..grader.templates import load_template, render_rubric_list
from ..rewards.recall import parse_coverage_judgments
from ._io import commit_jsonl, merge_manifest, read_jsonl
from .generate import select_subset
from .judge_client import Judge, check_health, judge_ident, resolve_judge

__all__ = ["TEMPLATE", "TYPES", "CUE_RE", "parse_type", "parse_types", "typing_prompts", "load_types",
           "pa_fields", "grader_fields", "cue_free_share"]

TEMPLATE = "eval/rubric_types"
TYPES = ("presence", "absence")
# A typing-sanity heuristic: wording that forbids or limits; an "absence" label
# on an item with no such cue is suspect.
CUE_RE = re.compile(r"\b(does not|do not|doesn't|must not|should not|never|avoids?|avoiding|without|"
                    r"refrains?|no |not |free of|only|contains no|rather than|instead of|limits?|exactly|"
                    r"at most|no more than)\b", re.IGNORECASE)


def parse_type(v) -> str | None:
    """The "type" field -> "presence" | "absence" | None. Case-insensitive,
    surrounding quotes / whitespace stripped, a "-based" suffix tolerated."""
    if not isinstance(v, str):
        return None
    s = v.strip().strip("\"'`").strip().lower()
    for t in TYPES:
        if s.startswith(t):
            return t
    return None


def parse_types(raw: str, n_rubrics: int) -> list[str | None]:
    """The typing JSON -> per-item types in rubric order. {"1": "presence"} or
    {"1": {"type": "presence"}} per item; a missing or invalid entry is None
    (never a guess). A body that is not a JSON object is a ParseError."""
    obj = loads_lenient(raw)
    if not isinstance(obj, dict):
        raise ParseError(f"rubric types must be a JSON object, got {type(obj).__name__}")
    out: list[str | None] = []
    for idx in range(1, n_rubrics + 1):
        val = obj.get(str(idx))
        if isinstance(val, dict):
            val = val.get("type")
        out.append(parse_type(val))
    return out


def typing_prompts(eval_cfg: dict) -> list[dict]:
    """The prompts of the domain's trajectory ∪ grader subsets with their
    rubrics, sorted by prompt_id."""
    rows = pq.read_table(eval_cfg["eval_pool"], columns=["prompt_id", "rubrics"]).to_pylist()
    chosen: dict[str, dict] = {}
    for name, spec in eval_cfg["subsets"].items():
        for row in select_subset(rows, int(spec["n"]), int(spec["seed"]), name):
            chosen[row["prompt_id"]] = row
    return sorted(chosen.values(), key=lambda r: r["prompt_id"])


async def _type_one(judge: Judge, template, sem: asyncio.Semaphore, row: dict) -> dict:
    texts = [x["text"] for x in row["rubrics"]]
    rec = {"prompt_id": row["prompt_id"], "n_rubrics": len(texts), "ok": False, "error": None,
           "finish_reason": None, "raw": "", "types": None,
           "weights": [float(x["weight"]) for x in row["rubrics"]]}
    async with sem:
        try:
            raw, finish = await judge.chat(template.messages(rubric_list_string=render_rubric_list(texts)))
            rec["raw"], rec["finish_reason"] = raw, finish
            if finish == "length":
                raise ValueError("truncated at max_tokens (finish_reason=length)")
            rec["types"] = parse_types(raw, len(texts))
            rec["ok"] = True
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def cue_free_share(records: list[dict], rows: list[dict]) -> dict:
    """Of the items typed absence, the share whose wording carries no negation / limit cue."""
    text_of = {r["prompt_id"]: [x["text"] for x in r["rubrics"]] for r in rows}
    n_abs = n_free = 0
    for rec in records:
        if not rec["ok"]:
            continue
        for t, text in zip(rec["types"], text_of.get(rec["prompt_id"], [])):
            if t == "absence":
                n_abs += 1
                n_free += not CUE_RE.search(text)
    return {"n_absence": n_abs, "n_absence_cue_free": n_free,
            "absence_cue_free_share": round(n_free / n_abs, 4) if n_abs else None}


async def run_pass(eval_cfg: dict, out: Path, judge_cfg: dict, config_path: str) -> int:
    rows = typing_prompts(eval_cfg)
    judge = Judge(judge_cfg)  # first: fails fast on missing replica URLs
    template = load_template(TEMPLATE)
    sem = asyncio.Semaphore(int(judge_cfg.get("concurrency", 32)))
    print(f"[rubric_types] {eval_cfg['domain']}: typing {len(rows)} prompts' rubrics via {judge_cfg['model']} -> {out}")
    try:
        records = await asyncio.gather(*(_type_one(judge, template, sem, r) for r in rows))
    finally:
        await judge.close()
    n_ok = sum(1 for r in records if r["ok"])
    counts = {"presence": 0, "absence": 0, None: 0}
    for r in records:
        if r["ok"]:
            for t in r["types"]:
                counts[t] += 1
    sanity = cue_free_share(records, rows)
    print(f"[rubric_types] {n_ok}/{len(records)} prompts typed; items: {counts['presence']} presence, "
          f"{counts['absence']} absence, {counts[None]} untyped; absence cue-free share {sanity['absence_cue_free_share']}")
    if n_ok < 0.98 * len(records):
        print("WARNING: >2% typing failures — inspect errors; artifact not committed")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    commit_jsonl(out, records)
    merge_manifest(out.parent, {"template": TEMPLATE, "judge": judge_ident(judge_cfg)},
                   "domains", eval_cfg["domain"],
                   {"artifact": out.name, "pool": eval_cfg["eval_pool"], "subsets": eval_cfg["subsets"],
                    "eval_cfg": config_path, "n_prompts": len(records), "n_ok": n_ok,
                    "n_presence": counts["presence"], "n_absence": counts["absence"],
                    "n_untyped_items": counts[None], **sanity})
    print(f"wrote {out} + manifest.json")
    return 0


# ---------------------------------------------------------------- the fields

def load_types(path: str | Path) -> dict[str, dict]:
    """The artifact -> {prompt_id: {"types": [...], "weights": [...]}} over its ok records."""
    out = {}
    for rec in read_jsonl(path):
        if rec.get("ok") and rec.get("types") is not None:
            out[rec["prompt_id"]] = {"types": rec["types"], "weights": rec["weights"]}
    return out


def pa_fields(items) -> dict:
    """Pooled presence / absence fields over (type, met, weight) items:
    count-based rates and shares, and their weight-based twins (`_w`, |w|).
    Items whose type is neither label are left out. None where undefined."""
    c = {"presence": [0, 0, 0.0, 0.0], "absence": [0, 0, 0.0, 0.0]}  # n, n_met, w, w_met
    for t, met, w in items:
        if t not in c:
            continue
        row = c[t]
        row[0] += 1
        row[2] += abs(w)
        if met:
            row[1] += 1
            row[3] += abs(w)
    p, a = c["presence"], c["absence"]

    def rate(num, den):
        return round(num / den, 4) if den else None

    return {
        "n_presence_items": p[0], "n_absence_items": a[0],
        "presence_met_rate": rate(p[1], p[0]), "absence_met_rate": rate(a[1], a[0]),
        "presence_share_met": rate(p[1], p[1] + a[1]), "presence_share_all": rate(p[0], p[0] + a[0]),
        "presence_met_rate_w": rate(p[3], p[2]), "absence_met_rate_w": rate(a[3], a[2]),
        "presence_share_met_w": rate(p[3], p[3] + a[3]), "presence_share_all_w": rate(p[2], p[2] + a[2]),
    }


def grader_fields(records: list[dict], types: dict[str, dict]) -> dict:
    """recall_scored.jsonl records (fork.eval.rubric_based) -> the pooled
    presence / absence satisfaction rates of one label."""
    items = []
    n_typed = n_untyped = n_unscored = 0
    for r in records:
        entry = types.get(r["prompt_id"])
        if entry is None or len(entry["types"]) != r["n_rubrics"]:
            n_untyped += 1
            continue
        try:
            verdicts = parse_coverage_judgments(r["raw"], r["n_rubrics"]) if r["ok"] else None
        except (TypeError, ValueError):
            verdicts = None
        if verdicts is None:
            n_unscored += 1
            continue
        n_typed += 1
        for i, (t, w) in enumerate(zip(entry["types"], entry["weights"]), start=1):
            items.append((t, verdicts.get(str(i), False), w))
    out = pa_fields(items)
    out.update(n_typed_responses=n_typed, n_untyped_responses=n_untyped, n_unscored_responses=n_unscored)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Type a domain's eval-subset rubrics presence / absence (once, with the training grader)")
    ap.add_argument("--config", required=True, help="configs/eval/<domain>.yaml")
    ap.add_argument("--out", default=None, help="artifact path (default outputs/rubric_types/<domain>.jsonl)")
    ap.add_argument("--base-urls", default=None, help="grader replica URLs (default: the eval config's local_judge.base_urls)")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the pass's size, call nothing")
    args = ap.parse_args()
    eval_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else Path("outputs/rubric_types") / f"{eval_cfg['domain']}.jsonl"
    rows = typing_prompts(eval_cfg)
    n_items = sum(len(r["rubrics"]) for r in rows)
    print(f"[rubric_types] {eval_cfg['domain']}: {len(rows)} prompts of {eval_cfg['eval_pool']}, "
          f"{n_items} rubric items, {len(rows)} grader calls -> {out}")
    if args.dry_run:
        return
    if out.exists():
        print(f"[rubric_types] {out} exists — skipped (delete it to redo the pass)")
        return
    judge_cfg = resolve_judge(eval_cfg, "local", args.base_urls, args.concurrency)
    check_health(judge_cfg["base_urls"])
    raise SystemExit(asyncio.run(run_pass(eval_cfg, out, judge_cfg, args.config)))


if __name__ == "__main__":
    main()
