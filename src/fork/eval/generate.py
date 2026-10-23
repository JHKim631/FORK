"""Generate a checkpoint's responses on an evaluation subset or an OOD pool.

Loads the eval pool, draws the seeded subset named by --subset from the eval
config (`grader`: the 100 prompts of the final-checkpoint evaluation;
`trajectory`: the 300 prompts scored at every checkpoint), and batch-generates
with vLLM offline — greedy decoding, the paper's 2,000-token response cap. The
policy sees ONLY the user prompt, no system message, matching the training-time
context. --model takes an HF model dir or hub id: the untrained base policy
(Qwen/Qwen2.5-7B-Instruct) and merged checkpoints go through the same path.

--pool <parquet> (the OOD pools of fork.data.ood) generates on EVERY row of the
pool instead of a subset draw, into ood_responses.jsonl; a pool row with
`messages` (HealthBench's conversation turns) is sent as those turns.

Output: <out-dir>/<label>/<subset>_responses.jsonl, one
{prompt_id, prompt, response, response_tokens} per line.

    python -m fork.eval.generate --config configs/eval/medical.yaml --model Qwen/Qwen2.5-7B-Instruct \\
        --label base_medical --subset grader
    python -m fork.eval.generate --config configs/eval/medical.yaml --pool data/healthbench/eval.parquet \\
        --model outputs/hf/fork_beta1.25_medical/step_291 --label fork_beta1.25_medical@291
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from ._io import commit_jsonl

__all__ = ["select_subset", "load_subset", "load_pool", "build_conversations"]


def select_subset(rows: list[dict], n: int, seed: int, salt: str) -> list[dict]:
    """Deterministic subset of the eval pool: sort by prompt_id (parquet order
    must not matter), then seeded sample. `salt` separates the trajectory and
    grader draws taken with the same seed."""
    ordered = sorted(rows, key=lambda r: r["prompt_id"])
    if n >= len(ordered):
        return ordered
    rng = random.Random(f"{seed}:{salt}")
    idx = sorted(rng.sample(range(len(ordered)), n))
    return [ordered[i] for i in idx]


def load_subset(cfg: dict, subset: str) -> list[dict]:
    spec = cfg["subsets"][subset]
    tbl = pq.read_table(cfg["eval_pool"], columns=["prompt_id", "prompt"])
    return select_subset(tbl.to_pylist(), int(spec["n"]), int(spec["seed"]), subset)


def load_pool(path: str) -> list[dict]:
    """Every row of an OOD pool, sorted by prompt_id; `messages` rides along
    when the pool has it (HealthBench)."""
    names = pq.read_schema(path).names
    cols = ["prompt_id", "prompt"] + (["messages"] if "messages" in names else [])
    return sorted(pq.read_table(path, columns=cols).to_pylist(), key=lambda r: r["prompt_id"])


def build_conversations(rows: list[dict]) -> list[list[dict]]:
    """Chat lists for llm.chat: the single user turn, or a row's own `messages`
    (a JSON list of turns — an OOD conversation pool). Never a system message."""
    convs = []
    for r in rows:
        convs.append(json.loads(r["messages"]) if r.get("messages") else [{"role": "user", "content": r["prompt"]}])
    return convs


def generate_responses(model: str, rows: list[dict], gen_cfg: dict, tensor_parallel_size: int) -> list[dict]:
    from vllm import LLM, SamplingParams  # lazy: heavy import, GPU-only path

    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, max_model_len=4096)
    params = SamplingParams(temperature=float(gen_cfg.get("temperature", 0.0)),
                            max_tokens=int(gen_cfg.get("max_tokens", 2000)))
    outputs = llm.chat(build_conversations(rows), params, use_tqdm=True)
    recs = []
    for row, out in zip(rows, outputs):
        completion = out.outputs[0]
        recs.append({"prompt_id": row["prompt_id"], "prompt": row["prompt"],
                     "response": completion.text, "response_tokens": len(completion.token_ids)})
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description="Greedy generation of one checkpoint's responses on an eval subset or OOD pool")
    ap.add_argument("--config", required=True, help="configs/eval/<domain>.yaml")
    ap.add_argument("--model", required=True, help="HF dir (a merged checkpoint) or hub id (the base policy)")
    ap.add_argument("--label", required=True, help="output name, e.g. base_medical or fork_beta1.25_medical@291")
    ap.add_argument("--subset", default=None, help="a subset of the eval config (grader | trajectory); or --pool")
    ap.add_argument("--pool", default=None, help="an OOD pool parquet: every row, output ood_responses.jsonl")
    ap.add_argument("--out-dir", default="outputs/eval")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if bool(args.subset) == bool(args.pool):
        raise SystemExit("give exactly one of --subset <name> or --pool <parquet>")
    if args.subset and args.subset not in cfg.get("subsets", {}):
        raise SystemExit(f"--subset {args.subset!r} not in {args.config}: have {sorted(cfg.get('subsets', {}))}")
    rows = load_pool(args.pool) if args.pool else load_subset(cfg, args.subset)
    what = f"pool {args.pool}" if args.pool else args.subset
    print(f"[generate:{args.label}] {len(rows)} prompts ({what}) with {args.model}")
    recs = generate_responses(args.model, rows, cfg.get("generation", {}), args.tensor_parallel_size)

    out = Path(args.out_dir) / args.label / ("ood_responses.jsonl" if args.pool else f"{args.subset}_responses.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    commit_jsonl(out, recs)   # atomic: a crashed generation leaves no responses file
    n_empty = sum(1 for r in recs if not r["response"].strip())
    print(f"wrote {out}  ({len(recs)} responses, {n_empty} empty)")


if __name__ == "__main__":
    main()
