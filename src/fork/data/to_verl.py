"""Convert a prepared parquet (data/<domain>/*.parquet) to verl's RLHFDataset format.

verl's dataset expects, per row:
    prompt        chat list [{"role": "user", "content": <question>}] — user-only, no
                  system message: the policy sees only the user prompt (verl applies
                  the tokenizer's chat template on top)
    data_source   a routing tag (unused by the reward, kept informative)
    reward_model  {"style": "rule", "ground_truth": <JSON string>}
    extra_info    {"index": int, "prompt_id": str}

ground_truth carries everything the reward needs to grade a rollout —
{prompt_id, prompt, rubrics: [{text, weight}], reference_answer} — as a JSON
string, because verl hands the reward function ground_truth per sample while
the raw dataset row stays behind. extra_info.prompt_id doubles as the GRPO
group key (all n rollouts of one prompt share it). The policy never sees the
rubrics.

    python -m fork.data.to_verl --data data/medical/train.parquet --out data/medical/verl/train.parquet
    python -m fork.data.to_verl --data data/medical/eval.parquet  --out data/medical/verl/eval.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["convert_row"]


def convert_row(row: dict, index: int, data_source: str) -> dict:
    gt = {
        "prompt_id": row["prompt_id"],
        "prompt": row["prompt"],
        "rubrics": [{"text": r["text"], "weight": float(r["weight"])}
                    for r in row["rubrics"]],
        "reference_answer": row.get("reference_answer") or "",
    }
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "data_source": data_source,
        "reward_model": {"style": "rule", "ground_truth": json.dumps(gt, ensure_ascii=False)},
        "extra_info": {"index": index, "prompt_id": row["prompt_id"]},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="prepared parquet -> verl RLHFDataset parquet")
    ap.add_argument("--data", required=True, help="fork.data.prepare / curate output parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-source", default=None,
                    help="the routing tag (default: the parquet's own `source` column)")
    args = ap.parse_args()

    rows = pq.read_table(args.data).to_pylist()
    tag = args.data_source or (rows[0].get("source") if rows else None) or "rubrichub"
    converted = [convert_row(r, i, tag) for i, r in enumerate(rows)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(converted), out)
    print(f"wrote {out}  ({len(converted)} rows, from {args.data})")


if __name__ == "__main__":
    main()
