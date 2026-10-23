"""Prepare a RubricHub slice (medical, science) as the train / eval parquets.

RubricHub (sojuL/RubricHub_v1) stores each domain as one parquet of prompts
with their rubric items and positive integer weights, and no reference
answers. The builder filters on rubric count, de-duplicates by prompt,
optionally drops prompts over the policy's token budget, shuffles with the
seed and splits into disjoint train / eval sets written in the schema every
downstream step reads (data/schemas.py):

    prompt_id, prompt, rubrics [{text, weight}], reference_answer, source

The shipped data/medical and data/science splits were built with the configs
under configs/data/ at the pinned RubricHub commit; the same command rebuilds
them byte for byte.

    python -m fork.data.prepare --config configs/data/medical.yaml
    python -m fork.data.prepare --config configs/data/science_raw.yaml   # then fork.data.curate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download

OUT_SCHEMA = pa.schema(
    [
        ("prompt_id", pa.string()),
        ("prompt", pa.string()),
        ("rubrics", pa.list_(pa.struct([("text", pa.string()), ("weight", pa.float64())]))),
        ("reference_answer", pa.string()),
        ("source", pa.string()),
    ]
)


def _prompt_text(prompt_field) -> str:
    """RubricHub 'prompt' is a list of {content, role} messages (a single user
    turn). Join contents so the policy sees exactly the user message."""
    if isinstance(prompt_field, str):
        return prompt_field.strip()
    parts = []
    for msg in prompt_field:
        content = (msg.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts).strip()


def _prompt_id(text: str, prefix: str = "med-") -> str:
    return prefix + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _hf_token_lengths(tokenizer_name: str):
    """verl's prompt-length measure (RLHFDataset.filter_overlong_prompts): the
    chat-templated token count with the generation prompt appended, user turn
    only. transformers is imported only when the budget key is set."""
    from transformers import AutoTokenizer  # lazy: a training-environment dependency

    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def lengths(texts: list[str]) -> list[int]:
        out: list[int] = []
        for i in range(0, len(texts), 2000):
            chats = [[{"role": "user", "content": x}] for x in texts[i:i + 2000]]
            ids = tok.apply_chat_template(chats, add_generation_prompt=True, tokenize=True)
            out.extend(len(x) for x in ids)
        return out

    return lengths


def _dist(values: list[int]) -> dict:
    """min / median / p99 / max summary for meta.json."""
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {"min": s[0], "median": s[n // 2], "p99": s[min(n - 1, int(n * 0.99))], "max": s[-1]}


def _rows_from_parquet(path: str | Path) -> list[dict]:
    tbl = pq.read_table(path, columns=["prompt", "Rubrics", "reward_model"])
    prompts = tbl.column("prompt").to_pylist()
    rubrics = tbl.column("Rubrics").to_pylist()
    rms = tbl.column("reward_model").to_pylist()
    rows = []
    for p, rub, rm in zip(prompts, rubrics, rms):
        text = _prompt_text(p)
        ref = (rm.get("ground_truth") or "").strip() if isinstance(rm, dict) else ""
        items = [
            {"text": (r["criterion"] or "").strip(), "weight": float(r["points"])}
            for r in rub
            if (r.get("criterion") or "").strip()
        ]
        rows.append({"prompt": text, "rubrics": items, "reference_answer": ref})
    return rows


def prepare(config: dict, token_len_fn=None, source: str | Path | None = None) -> dict:
    """Build the split. `source` is a local copy of the RubricHub parquet (the
    default downloads it at the config's pinned revision); `token_len_fn`
    (texts -> token lengths) replaces the HF tokenizer when `max_prompt_tokens`
    is set (an offline seam)."""
    path = Path(source) if source else Path(hf_hub_download(
        config["hf_repo"], config["hf_file"], repo_type=config.get("repo_type", "dataset"),
        revision=config.get("revision")))
    raw = _rows_from_parquet(path)
    n_raw = len(raw)

    lo, hi = config["min_rubrics"], config["max_rubrics"]
    prefix = config.get("id_prefix", "med-")
    src = config.get("source", "rubrichub_medical")
    seen: set[str] = set()
    kept: list[dict] = []
    n_countfilt = n_dup = n_empty = 0
    for row in raw:
        if not row["prompt"]:
            n_empty += 1
            continue
        if not (lo <= len(row["rubrics"]) <= hi):
            n_countfilt += 1
            continue
        pid = _prompt_id(row["prompt"], prefix)
        if pid in seen:
            n_dup += 1
            continue
        seen.add(pid)
        row["prompt_id"] = pid
        row["source"] = src
        kept.append(row)

    # Prompt-token budget. Absent key = no filter.
    n_overlong = 0
    max_tok = config.get("max_prompt_tokens")
    if max_tok is not None:
        lengths = (token_len_fn or _hf_token_lengths(config["tokenizer"]))(
            [r["prompt"] for r in kept])
        for row, n_tok in zip(kept, lengths):
            row["_prompt_tokens"] = n_tok
        n_before = len(kept)
        kept = [r for r in kept if r["_prompt_tokens"] <= int(max_tok)]
        n_overlong = n_before - len(kept)

    rng = random.Random(config["seed"])
    rng.shuffle(kept)

    n_eval, n_train = config["n_eval"], config["n_train"]
    if len(kept) < n_eval + n_train:
        raise ValueError(
            f"only {len(kept)} usable rows after filtering; need {n_eval + n_train}"
        )
    eval_rows = kept[:n_eval]
    train_rows = kept[n_eval : n_eval + n_train]

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "eval.parquet", eval_rows)
    _write(out_dir / "train.parquet", train_rows)

    meta = {
        "config": config,
        "source_file": path.name,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_raw": n_raw,
        "n_dropped_empty": n_empty,
        "n_dropped_rubric_count": n_countfilt,
        "n_dropped_duplicate": n_dup,
        "n_dropped_overlong": n_overlong,   # 0 unless max_prompt_tokens is set
        "n_usable": len(kept),
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        # The distributions the grader's token caps and the policy's prompt
        # budget are sized against.
        "rubric_count": {
            "train": _dist([len(r["rubrics"]) for r in train_rows]),
            "eval": _dist([len(r["rubrics"]) for r in eval_rows]),
        },
        "prompt_chars": {
            "train": _dist([len(r["prompt"]) for r in train_rows]),
            "eval": _dist([len(r["prompt"]) for r in eval_rows]),
        },
        **({"prompt_tokens": {
                "max_prompt_tokens": int(max_tok),
                "train": _dist([r["_prompt_tokens"] for r in train_rows]),
                "eval": _dist([r["_prompt_tokens"] for r in eval_rows]),
            }} if max_tok is not None else {}),
        "note": "RubricHub carries no reference answers; reference_answer is empty.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def _write(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt"],
                "rubrics": r["rubrics"],
                "reference_answer": r["reference_answer"],
                "source": r["source"],
            }
            for r in rows
        ],
        schema=OUT_SCHEMA,
    )
    pq.write_table(table, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", default=None, help="a local copy of the RubricHub parquet (default: download)")
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    meta = prepare(config, source=args.source)
    print(json.dumps({k: v for k, v in meta.items() if k != "config"}, indent=2, default=str))


if __name__ == "__main__":
    main()
