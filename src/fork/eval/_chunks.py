"""Resumable judge calls for the evaluation modules.

A lane file (rubric_based.*/recall_scored.jsonl, rubric_free.*/vs_<y>.jsonl, ...)
is committed whole and only after its parse gate — the file guards depend on
that. So that an interrupted run loses nothing, a module keeps a per-file
CACHE of the calls already answered ok — <target>.cache.jsonl beside the
target — and a rerun answers only what is missing, then assembles cache +
fresh records in the canonical order, applies the gate, commits the file and
removes the cache. A failed call is never cached, so the next run retries it.
A stop signal (SIGTERM / SIGINT) starts no new call; the ones in flight finish
and are cached; the exit is PARTIAL.
"""

from __future__ import annotations

import asyncio
import json
import signal
from pathlib import Path
from typing import Awaitable, Callable, Hashable

__all__ = ["PARTIAL", "CallCache", "install_stop", "run_chunk"]

PARTIAL = 3   # exit status: calls still unanswered, the cache holds the rest — rerun to continue


class CallCache:
    """The ok answers of one target file, keyed by `key_fields` of the record;
    lives beside the target as <stem>.cache.jsonl."""

    def __init__(self, target: str | Path, key_fields: tuple[str, ...] = ()):
        target = Path(target)
        self.path = target.with_name(target.stem + ".cache.jsonl")
        self.key_fields = key_fields

    def key(self, rec: dict) -> tuple:
        return tuple(rec[k] for k in self.key_fields)

    def load(self) -> dict[tuple, dict]:
        out: dict[tuple, dict] = {}
        if not self.path.exists():
            return out
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # a line cut by a kill mid-write: that call is simply redone
            out[self.key(rec)] = rec
        return out

    def append(self, rec: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    def discard(self) -> None:
        if self.path.exists():
            self.path.unlink()


def install_stop() -> asyncio.Event:
    """An Event set by SIGTERM / SIGINT; call inside the running loop."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError, ValueError):   # no signal support / not the main thread
            pass
    return stop


async def run_chunk(items: list[tuple[Hashable, object]], make_call: Callable[[object], Awaitable[dict]],
                    concurrency: int, stop: asyncio.Event,
                    on_done: Callable[[Hashable, dict], object]) -> dict:
    """make_call(payload) for each (key, payload) with `concurrency` workers;
    on_done(key, record) as each lands. Once `stop` is set no new call starts;
    the ones in flight finish. Returns {key: record} for the calls attempted."""
    queue = list(items)
    results: dict = {}

    async def worker() -> None:
        while queue and not stop.is_set():
            key, payload = queue.pop(0)
            rec = await make_call(payload)
            results[key] = rec
            on_done(key, rec)

    await asyncio.gather(*(worker() for _ in range(max(1, min(concurrency, len(items))))))
    return results
