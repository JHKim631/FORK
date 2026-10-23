"""Atomic artifact commit for the evaluation pipeline.

Every evaluation step is guarded by the existence of its output file
(scripts/eval.sh). Writing that file in place would let a crash mid-write — or a
grader that fails its parse gate after streaming — leave a partial artifact
the guard silently accepts next time. commit_jsonl writes a sibling ``.tmp``
and renames it into place, so the guarded path only ever appears complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

__all__ = ["commit_jsonl", "merge_manifest", "read_jsonl"]

ROOT = Path(__file__).resolve().parents[3]


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def commit_jsonl(path: str | Path, records: list[dict]) -> None:
    """Write ``records`` as JSONL to ``path`` atomically (same-directory rename)."""
    path = Path(path)
    tmp = path.parent / (path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def merge_manifest(lane_dir: str | Path, header: dict, group: str, key: str, entry: dict) -> Path:
    """Provenance for one lane directory: create or update ``<lane_dir>/manifest.json``
    — the header fields (lane, judge, template, ...) are set, and ``entry`` is
    stored under ``[group][key]`` stamped with the time, the git sha and the
    template file's sha256 when the header names a template under prompts/.
    Best-effort provenance, never a guard: the guarded artifacts are the jsonl
    files beside it."""
    lane_dir = Path(lane_dir)
    path = lane_dir / "manifest.json"
    m = json.loads(path.read_text()) if path.exists() else {}
    m.update(header)
    tpl = header.get("template")
    if isinstance(tpl, str):
        tpath = ROOT / "prompts" / f"{tpl}.txt"
        if tpath.exists():
            m["template_sha256"] = hashlib.sha256(tpath.read_bytes()).hexdigest()
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=ROOT).stdout.strip() or None
    except OSError:
        sha = None
    m.setdefault(group, {})[key] = {**entry, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_sha": sha}
    path.write_text(json.dumps(m, indent=2))
    return path
