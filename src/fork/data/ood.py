"""The two out-of-domain benchmark pools: HealthBench for the medical arms,
ResearchQA for the science arms — each a parquet in the eval-pool shape
(prompt_id, prompt, rubrics [{text, weight, ...}], source), so fork.eval.generate
and fork.eval.rubric_free read it like an in-domain pool, plus what the
benchmark's own grader (fork.eval.ood) needs.

HealthBench (Arora et al. 2025; OpenAI simple-evals): the public oss_eval file
— 5,000 conversations, sha256 pinned — sampled EXACTLY as the official script
does, random.Random(0).sample(examples, n) over the file's order. A row keeps
the official prompt_id, the conversation turns (`messages`, a JSON list; every
one ends with a user turn) and, as `prompt`, the turns rendered "role: content"
joined by blank lines (the official convo_str without the response; the
rubric-free judge's {question}); `rubrics` = [{text: criterion, weight:
points, tags}] — points may be NEGATIVE; the canonical score divides the met
points by the positive total.

ResearchQA (Li et al. 2026; HF realliyifei/ResearchQA at a pinned revision):
the test split (3,750 = 50 per field) filtered to the two natural-science
domains ("Life & Earth Sciences", "Physical & Theoretical Sciences" — 1,100
queries), sorted by id, random.Random(0).sample(..., n). A row keeps the
official id, `prompt` = the query, `rubrics` = [{text: rubric_item, weight:
1.0, types: [...]}] (the benchmark has no weights; the canonical coverage is
unweighted) and the domain columns.

Both writers record the source's identity (URL / repo, file, sha256, revision)
and the draw in <out-dir>/meta.json. The shipped data/healthbench and
data/researchqa are these pools; the pins here are what makes a rebuild identical.

    python -m fork.data.ood --bench healthbench    # -> data/healthbench/eval.parquet + meta.json
    python -m fork.data.ood --bench researchqa     # -> data/researchqa/eval.parquet + meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["HB_URL", "HB_FILE", "HB_SHA256", "RQ_REPO", "RQ_REVISION", "RQ_FILE", "RQ_SHA256",
           "RQ_DOMAINS", "render_conversation", "healthbench_rows", "researchqa_rows",
           "build", "sha256_of"]

ROOT = Path(__file__).resolve().parents[3]

HB_URL = "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"
HB_FILE = "2025-05-07-06-14-12_oss_eval.jsonl"
HB_SHA256 = "e99dd3c6372c10d6fcc5e385c5fae69d0dd40392dae56836ef9493ae324ecd2f"   # 5,000 examples

RQ_REPO = "realliyifei/ResearchQA"
RQ_REVISION = "bf8a4cfef073ecfc0275c57acf8ca960e4dc79d6"
RQ_FILE = "data/test-00000-of-00001.parquet"
RQ_SHA256 = "10838fd1c6228884392362dce0ad9a13e8b0dce796f01d61221c4ce2e989bce9"   # 3,750 rows
RQ_DOMAINS = ("Life & Earth Sciences", "Physical & Theoretical Sciences")

DEFAULT_N = 100
DEFAULT_SEED = 0


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_conversation(messages: list[dict]) -> str:
    """simple-evals' convo_str: "role: content" turns joined by blank lines."""
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def healthbench_rows(examples: list[dict], n: int = DEFAULT_N, seed: int = DEFAULT_SEED) -> list[dict]:
    """The official draw (random.Random(seed).sample over the file's order) as pool rows."""
    sample = random.Random(seed).sample(examples, n) if n < len(examples) else list(examples)
    return [{
        "prompt_id": e["prompt_id"],
        "prompt": render_conversation(e["prompt"]),
        "messages": json.dumps(e["prompt"], ensure_ascii=False),
        "rubrics": [{"text": r["criterion"], "weight": float(r["points"]), "tags": list(r.get("tags") or [])}
                    for r in e["rubrics"]],
        "example_tags": list(e.get("example_tags") or []),
        "source": "healthbench_oss_eval",
    } for e in sample]


def researchqa_rows(rows: list[dict], n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
                    domains: tuple[str, ...] = RQ_DOMAINS) -> list[dict]:
    """The natural-science test queries, sorted by id, then the seeded draw, as pool rows."""
    pool = sorted((r for r in rows if r["general_domain"] in domains), key=lambda r: r["id"])
    sample = random.Random(seed).sample(pool, n) if n < len(pool) else pool
    return [{
        "prompt_id": r["id"],
        "prompt": r["query"],
        "rubrics": [{"text": it["rubric_item"], "weight": 1.0, "types": list(it.get("type") or [])}
                    for it in r["rubric"]],
        "general_domain": r["general_domain"], "subdomain": r["subdomain"], "field": r["field"],
        "source": "researchqa_test",
    } for r in sample]


def _healthbench_source(source: Path | None, out_dir: Path) -> Path:
    path = Path(source) if source else out_dir / HB_FILE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[ood] downloading {HB_URL} -> {path}")
        urllib.request.urlretrieve(HB_URL, path)
    return path


def _researchqa_source(source: Path | None) -> Path:
    if source:
        return Path(source)
    from huggingface_hub import hf_hub_download  # lazy: the cache lives under $HF_HOME

    return Path(hf_hub_download(RQ_REPO, RQ_FILE, repo_type="dataset", revision=RQ_REVISION))


def build(bench: str, out_dir: Path, n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
          source: Path | None = None) -> tuple[Path, dict]:
    """Write <out_dir>/eval.parquet + meta.json for one bench; refuses a source
    whose sha256 is not the pinned one. Returns (parquet path, meta)."""
    out_dir = Path(out_dir)
    if bench == "healthbench":
        path = _healthbench_source(source, out_dir)
        sha, pinned = sha256_of(path), HB_SHA256
        origin = {"url": HB_URL, "file": path.name}
    elif bench == "researchqa":
        path = _researchqa_source(source)
        sha, pinned = sha256_of(path), RQ_SHA256
        origin = {"hf_repo": RQ_REPO, "revision": RQ_REVISION, "file": RQ_FILE}
    else:
        raise ValueError(f"bench must be healthbench | researchqa, got {bench!r}")
    if sha != pinned:
        raise SystemExit(f"{path}: sha256 {sha} != the pinned {pinned} — not the file the pool was defined on")
    if bench == "healthbench":
        examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows, n_source, n_candidates = healthbench_rows(examples, n, seed), len(examples), len(examples)
    else:
        src_rows = pq.read_table(path).to_pylist()
        rows = researchqa_rows(src_rows, n, seed)
        n_source = len(src_rows)
        n_candidates = sum(1 for r in src_rows if r["general_domain"] in RQ_DOMAINS)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "eval.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    meta = {
        "bench": bench, "source": {**origin, "sha256": sha}, "n": len(rows), "seed": seed,
        "draw": "random.Random(seed).sample" + (" over the file's order" if bench == "healthbench"
                                               else f" over the {' + '.join(RQ_DOMAINS)} test rows sorted by id"),
        "n_source_rows": n_source, "n_candidates": n_candidates,
        "n_items": sum(len(r["rubrics"]) for r in rows),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if bench == "healthbench":
        meta["n_negative_items"] = sum(1 for r in rows for it in r["rubrics"] if it["weight"] < 0)
        meta["turns_per_conversation"] = {str(k): v for k, v in sorted(
            Counter(len(json.loads(r["messages"])) for r in rows).items())}
    else:
        meta["domains"] = list(RQ_DOMAINS)
        meta["n_citation_items"] = sum(1 for r in rows for it in r["rubrics"] if "Citation" in it["types"])
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return parquet, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an OOD benchmark pool (HealthBench | ResearchQA)")
    ap.add_argument("--bench", required=True, choices=["healthbench", "researchqa"])
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--source", default=None, help="the source file (default: download / the HF cache)")
    ap.add_argument("--out-dir", default=None, help="default data/<bench>/")
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data" / args.bench
    parquet, meta = build(args.bench, out_dir, args.n, args.seed, args.source)
    print(f"wrote {parquet} ({meta['n']} rows, {meta['n_items']} rubric items) + meta.json")


if __name__ == "__main__":
    main()
