"""The rubric-free evaluation (paper Section 4.1): pairwise judging of two
models' responses to the same prompts, in BOTH presentation orders.

Two judge prompts (configs/eval/<domain>.yaml `rubric_free_prompts`):
  p1  the five-dimension Likert judge of Mahmoud et al. (prompts/eval/rubric_free_judge.txt,
      the science twin for the science pools): scores on completeness, factual
      correctness, conciseness, relevance, safety and overall, 1–7 each. The
      harness averages the two orders per prompt; the overall delta decides the
      winner (fork.eval.report). The paper's win rates and Table 8.
  p2  the MT-Bench pair-v2 judge (prompts/eval/rubric_free_judge_mtbench.txt):
      one verdict [[A]] / [[B]] / [[C]] after an explanation; a win needs both
      orders to agree, anything else is a tie. The paper's Table 4.

Records are written per (prompt, order) with the A/B results already mapped
back to the x / y sides:
  likert   {prompt_id, order: "xy"|"yx", ok, error, raw, x: {dim: int}, y: {dim: int}}
  verdict  {prompt_id, order, ok, error, raw, verdict: "x"|"y"|"tie"}

    python -m fork.eval.rubric_free --config configs/eval/medical.yaml --prompt p1 \\
        --x-label fork_beta1.25_medical@291 --x outputs/eval/fork_beta1.25_medical@291/grader_responses.jsonl \\
        --y-label base_medical --y outputs/eval/base_medical/grader_responses.jsonl \\
        --out-file outputs/eval/fork_beta1.25_medical@291/rubric_free.gpt54.p1/vs_base_medical.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

import yaml

from ..grader.parsing import loads_lenient
from ..grader.templates import load_template
from ._chunks import PARTIAL, CallCache, install_stop, run_chunk
from ._io import commit_jsonl, merge_manifest, read_jsonl
from .judge_client import Judge, add_judge_args, judge_ident, resolve_judge

__all__ = ["DIMENSIONS", "FORMATS", "parse_judgment", "scores_to_xy", "parse_verdict", "verdict_to_xy"]

DIMENSIONS = ("completeness", "factual_correctness", "conciseness", "relevance", "safety", "overall")
FORMATS = ("likert", "verdict")
_VERDICT_RE = re.compile(r"\[\[([ABC])\]\]")


def parse_judgment(raw: str, dimensions=DIMENSIONS) -> dict:
    """Judge JSON -> {"A": {dim: int}, "B": {dim: int}}. Raises on any missing
    dimension or non-integer score — a partial judgment must not average in."""
    obj = loads_lenient(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"judgment must be a JSON object, got {type(obj).__name__}")
    out: dict[str, dict[str, int]] = {}
    for side, key in (("A", "response_A"), ("B", "response_B")):
        block = obj.get(key)
        if not isinstance(block, dict):
            raise ValueError(f"missing {key} block")
        scores = {}
        for dim in dimensions:
            cell = block.get(dim)
            score = cell.get("score") if isinstance(cell, dict) else cell
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(f"{key}.{dim}: no numeric score")
            score = int(score)
            if not 1 <= score <= 7:
                raise ValueError(f"{key}.{dim}: score {score} outside 1-7")
            scores[dim] = score
        out[side] = scores
    return out


def scores_to_xy(parsed: dict, order: str) -> tuple[dict, dict]:
    """Map A/B scores back to (x, y) given which order was presented."""
    if order == "xy":
        return parsed["A"], parsed["B"]
    if order == "yx":
        return parsed["B"], parsed["A"]
    raise ValueError(f"order must be 'xy' or 'yx', got {order!r}")


def parse_verdict(raw: str) -> str:
    """MT-Bench verdict -> "A" | "B" | "C" (tie): the LAST "[[A]]" / "[[B]]" /
    "[[C]]" in the body. The prompt asks for the verdict AFTER the explanation,
    and an explanation may quote the format, so the last token is the
    decision. A body with no verdict token raises — it must not count."""
    found = _VERDICT_RE.findall(raw)
    if not found:
        raise ValueError("no [[A]] / [[B]] / [[C]] verdict in the judgment")
    return found[-1]


def verdict_to_xy(verdict: str, order: str) -> str:
    """"A" / "B" / "C" under the presented order -> "x" | "y" | "tie"."""
    if order not in ("xy", "yx"):
        raise ValueError(f"order must be 'xy' or 'yx', got {order!r}")
    if verdict == "C":
        return "tie"
    if verdict not in ("A", "B"):
        raise ValueError(f"verdict must be A, B or C, got {verdict!r}")
    if order == "xy":
        return "x" if verdict == "A" else "y"
    return "y" if verdict == "A" else "x"


def _load_responses(path: str) -> dict[str, dict]:
    return {rec["prompt_id"]: rec for rec in read_jsonl(path)}


async def _judge_one(judge, template, sem, prompt_id: str, question: str,
                     x_resp: str, y_resp: str, order: str, fmt: str = "likert",
                     dimensions=DIMENSIONS) -> dict:
    resp_a, resp_b = (x_resp, y_resp) if order == "xy" else (y_resp, x_resp)
    rec = {"prompt_id": prompt_id, "order": order, "ok": False, "error": None, "raw": ""}
    rec.update({"verdict": None} if fmt == "verdict" else {"x": None, "y": None})
    async with sem:
        try:
            raw, _finish = await judge.chat(template.messages(
                question=question, response_A=resp_a, response_B=resp_b))
            rec["raw"] = raw
            if fmt == "verdict":
                rec["verdict"] = verdict_to_xy(parse_verdict(raw), order)
            else:
                parsed = parse_judgment(raw, dimensions)  # truncated JSON fails here -> error
                rec["x"], rec["y"] = scores_to_xy(parsed, order)
            rec["ok"] = True
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def prompt_spec(eval_cfg: dict, prompt: str) -> tuple[str, str]:
    """(template name, format) of a judge prompt id from the eval config."""
    specs = eval_cfg.get("rubric_free_prompts") or {}
    if prompt not in specs:
        raise SystemExit(f"--prompt {prompt!r} is not in the eval config's rubric_free_prompts ({', '.join(specs)})")
    spec = specs[prompt]
    fmt = spec.get("format", "likert")
    if fmt not in FORMATS:
        raise SystemExit(f"prompt {prompt}: format must be {' | '.join(FORMATS)}, got {fmt!r}")
    return spec["template"], fmt


async def main_async(args) -> int:
    eval_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    judge_cfg = resolve_judge(eval_cfg, args.judge, args.base_urls, args.concurrency)
    judge = Judge(judge_cfg)  # first: fail fast on a missing key / replica URLs
    name, fmt = prompt_spec(eval_cfg, args.prompt)
    template = load_template(name)
    xs, ys = _load_responses(args.x), _load_responses(args.y)
    shared = sorted(set(xs) & set(ys))
    if not shared:
        raise SystemExit("no shared prompt_ids between the two response files")
    dropped = (len(xs) - len(shared), len(ys) - len(shared))
    print(f"[rubric_free] {args.x_label} vs {args.y_label}: {len(shared)} prompts x 2 orders "
          f"via {judge_cfg['model']}, {name} ({fmt}) (dropped unshared: x={dropped[0]}, y={dropped[1]})")

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = [(pid, order) for pid in shared for order in ("xy", "yx")]   # the canonical record order
    cache = CallCache(out, ("prompt_id", "order"))
    cached = cache.load()
    pending = [k for k in keys if k not in cached]
    if cached:
        print(f"[rubric_free] {len(cached)} judgments cached, {len(pending)} pending")
    conc = int(judge_cfg.get("concurrency", 8))
    sem = asyncio.Semaphore(conc)
    stop = install_stop()

    def call(key):
        pid, order = key
        q = xs[pid].get("prompt") or ys[pid].get("prompt") or ""
        return _judge_one(judge, template, sem, pid, q, xs[pid]["response"], ys[pid]["response"], order, fmt)

    try:
        fresh = await run_chunk([(k, k) for k in pending], call, conc, stop,
                                lambda _k, rec: cache.append(rec) if rec["ok"] else None)
    finally:
        await judge.close()
    answered = {**cached, **fresh}
    records = [answered.get(k) for k in keys]
    if any(rec is None for rec in records):
        n_have = sum(rec is not None for rec in records)
        print(f"[rubric_free] partial: {n_have}/{len(keys)} judged" + (" (stopped)" if stop.is_set() else "")
              + f" — cache {cache.path}; rerun to continue")
        return PARTIAL

    n_ok = sum(1 for r in records if r["ok"])
    if n_ok < 0.98 * len(records):
        print(f"WARNING: >2% judge failures — inspect errors before aggregating ({out.name} not committed; "
              f"ok judgments stay cached in {cache.path})")
        return 1
    commit_jsonl(out, records)
    print(f"wrote {out}  ({n_ok}/{len(records)} judgments ok)")
    merge_manifest(out.parent, {"lane": out.parent.name, "judge": judge_ident(judge_cfg), "template": name,
                                "format": fmt, "dimensions": list(DIMENSIONS) if fmt == "likert" else None,
                                "eval_cfg": args.config},
                   "pairs", out.name, {"x": args.x_label, "y": args.y_label, "n": len(records), "n_ok": n_ok,
                                       "from_cache": len(cached)})
    cache.discard()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Pairwise rubric-free judging of two models' responses (one pair)")
    ap.add_argument("--config", required=True, help="configs/eval/<domain>.yaml")
    ap.add_argument("--prompt", default="p1", help="the judge prompt id (p1 Likert, p2 MT-Bench verdict)")
    ap.add_argument("--x-label", required=True)
    ap.add_argument("--y-label", required=True)
    ap.add_argument("--x", required=True, help="responses jsonl for x")
    ap.add_argument("--y", required=True, help="responses jsonl for y")
    ap.add_argument("--out-file", required=True, help="e.g. outputs/eval/<x>/rubric_free.<judge>.<prompt>/vs_<y>.jsonl")
    add_judge_args(ap)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
