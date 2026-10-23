"""Curate a prepared RubricHub split by prompt SOURCE.

RubricHub-Science's prompts are verbatim questions from RaR-Science, ResearchQA
and MegaScience, but the parquet carries no per-row source. Every prompt of an
already-prepared split (fork.data.prepare's output) is therefore attributed by
exact text match, whitespace collapsed, against the source datasets at pinned
revisions. A prompt matching only `keep` sources is kept (one found in several
kept sources included); a prompt matching only other sources is dropped; a
prompt matching no source, or a kept AND another source, is an error — its
attribution would be a guess.

The eval split keeps every kept row; the train split is subsampled to `n_train`
with `seed`, in the base split's order. Kept rows are copied unchanged (same
prompt_id, rubrics, schema), so every filter the base build applied still
holds. Writes `<output_dir>/{train,eval}.parquet`, `meta.json` (counts by
source) and `sources.json` (prompt_id -> the sources it matched, for every
base row).

    python -m fork.data.curate --config configs/data/science.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from huggingface_hub import hf_hub_download

from .prepare import _dist, _write

SPLITS = ("train", "eval")
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _source_texts(spec: dict):
    """Every text in the source's `column`, over its listed files at the pinned revision."""
    for f in spec["files"]:
        path = hf_hub_download(spec["hf_repo"], f, repo_type="dataset", revision=spec["revision"])
        for batch in pq.ParquetFile(path).iter_batches(columns=[spec["column"]], batch_size=100_000):
            yield from batch.column(spec["column"]).to_pylist()


def attribute(prompts: list[str], sources: dict, texts_fn=_source_texts) -> list[set[str]]:
    """For each prompt, the names of the sources containing it (whitespace collapsed)."""
    hits: dict[str, set[str]] = {_norm(p): set() for p in prompts}
    for name, spec in sources.items():
        for text in texts_fn(spec):
            found = hits.get(_norm(text)) if text else None
            if found is not None:
                found.add(name)
    return [hits[_norm(p)] for p in prompts]


def curate(config: dict, texts_fn=_source_texts) -> dict:
    """`texts_fn` (source spec -> texts) replaces the HF download (an offline seam)."""
    base, out = Path(config["base_dir"]), Path(config["output_dir"])
    keep = set(config["keep"])
    rows = {s: pq.read_table(base / f"{s}.parquet").to_pylist() for s in SPLITS}
    every = rows["train"] + rows["eval"]
    matched = dict(zip((r["prompt_id"] for r in every),
                       attribute([r["prompt"] for r in every], config["sources"], texts_fn)))
    unmatched = [pid for pid, m in matched.items() if not m]
    if unmatched:
        raise ValueError(f"{len(unmatched)} prompts match no source, e.g. {unmatched[:3]}")
    mixed = [pid for pid, m in matched.items() if m & keep and m - keep]
    if mixed:
        raise ValueError(f"{len(mixed)} prompts match a kept AND another source, e.g. {mixed[:3]}")

    kept = {s: [r for r in rows[s] if matched[r["prompt_id"]] <= keep] for s in SPLITS}
    n_train = int(config["n_train"])
    if len(kept["train"]) < n_train:
        raise ValueError(f"only {len(kept['train'])} kept train rows; need {n_train}")
    idx = sorted(random.Random(config["seed"]).sample(range(len(kept["train"])), n_train))
    final = {"train": [kept["train"][i] for i in idx], "eval": kept["eval"]}

    out.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        _write(out / f"{s}.parquet", final[s])
    (out / "sources.json").write_text(
        "{\n" + ",\n".join(f"{json.dumps(pid)}: {json.dumps(sorted(m))}"
                           for pid, m in matched.items()) + "\n}\n")

    def by_source(rs):
        return dict(Counter("+".join(sorted(matched[r["prompt_id"]])) for r in rs).most_common())

    meta = {
        "config": config,
        "n_base": {s: len(rows[s]) for s in SPLITS},
        "n_kept": {s: len(kept[s]) for s in SPLITS},
        "n_train": len(final["train"]),
        "n_eval": len(final["eval"]),
        "base_by_source": {s: by_source(rows[s]) for s in SPLITS},
        "by_source": {s: by_source(final[s]) for s in SPLITS},
        "rubric_count": {s: _dist([len(r["rubrics"]) for r in final[s]]) for s in SPLITS},
        "prompt_chars": {s: _dist([len(r["prompt"]) for r in final[s]]) for s in SPLITS},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    meta = curate(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    print(json.dumps({k: v for k, v in meta.items() if k != "config"}, indent=2, default=str))


if __name__ == "__main__":
    main()
