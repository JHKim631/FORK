"""The judge behind an evaluation lane, in one of two identities:

- ``gpt54`` — the reference grader: GPT-5.4 through the OpenAI API, with the
  call parameters of the eval config's `judge` block (temperature omitted when
  null; `max_completion_tokens`; an optional `reasoning_effort`). Needs
  OPENAI_API_KEY (the .env file). An OOD benchmark's own grader settings
  (`ood.grader_overrides.gpt54` in the eval config) are applied on top when
  the caller names the bench.
- ``local`` — the training grader's identity on the served replicas
  (scripts/serve_grader.sh): the grader config's model, temperature 0, its
  extra_body (thinking off), its recall token cap, round-robin over the
  replica URLs, and ONE 1.5x retry on a body cut at max_tokens. Never a remote
  provider. Used for the evaluations that isolate the reward formulation from
  the stronger reference grader (paper Figure 3) and for the trajectory
  (Figure 1).

Every lane module builds its Judge FIRST, so a missing key or replica fails
before any file or GPU work.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

__all__ = ["JUDGES", "Judge", "resolve_judge", "judge_ident", "check_health", "add_judge_args"]

JUDGES = ("gpt54", "local")
ROOT = Path(__file__).resolve().parents[3]


def _openai_client(judge_cfg: dict):
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set — put it in .env or export it first.")
    from openai import AsyncOpenAI  # lazy: keeps the module importable without the key

    return AsyncOpenAI(base_url=judge_cfg.get("base_url"), api_key=os.environ["OPENAI_API_KEY"],
                       timeout=float(judge_cfg.get("timeout_s", 180)), max_retries=3)


async def _openai_chat(client, judge_cfg: dict, messages: list[dict]) -> tuple[str, str | None]:
    """One chat call; returns (text, finish_reason) so callers can refuse
    truncated bodies instead of parsing deflated scores."""
    kwargs = {"model": judge_cfg["model"], "messages": messages,
              "max_completion_tokens": int(judge_cfg.get("max_completion_tokens", 4000))}
    if judge_cfg.get("temperature") is not None:
        kwargs["temperature"] = float(judge_cfg["temperature"])
    if judge_cfg.get("reasoning_effort"):
        kwargs["reasoning_effort"] = judge_cfg["reasoning_effort"]
    resp = await client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    return choice.message.content or "", choice.finish_reason


async def _local_chat(client, judge_cfg: dict, messages: list[dict]) -> tuple[str, str | None]:
    """One chat call in the training grader's identity: temperature 0,
    `max_tokens` and its extra_body — the request the training client sends.
    A body cut at max_tokens is retried ONCE at 1.5x; the caller refuses a
    body still truncated after that."""
    tokens = int(judge_cfg.get("max_tokens", 5000))
    extra = judge_cfg.get("extra_body") or None
    for attempt in (0, 1):
        resp = await client.chat.completions.create(
            model=judge_cfg["model"], messages=messages,
            temperature=float(judge_cfg.get("temperature", 0.0)),
            max_tokens=tokens, extra_body=extra)
        choice = resp.choices[0]
        text, finish = choice.message.content or "", choice.finish_reason
        if finish != "length" or attempt:
            return text, finish
        tokens = int(tokens * 1.5)
    raise RuntimeError("unreachable")  # pragma: no cover


class Judge:
    """The judge behind one lane: `chat(messages) -> (text, finish_reason)` and
    `close()`. `clients` injects the API clients — any objects with
    `.chat.completions.create` and `.close()`; by default they are built from
    the config."""

    def __init__(self, judge_cfg: dict, clients: list | None = None):
        self.cfg = judge_cfg
        self.kind = judge_cfg.get("kind", "openai")
        if self.kind not in ("openai", "local"):
            raise SystemExit(f"unknown judge kind {self.kind!r} (openai | local)")
        if clients is not None:
            self._clients = list(clients)
        elif self.kind == "openai":
            self._clients = [_openai_client(judge_cfg)]
        else:
            urls = [u for u in (judge_cfg.get("base_urls") or []) if u]
            if not urls:
                raise SystemExit("local judge: no replica URLs (pass --base-urls or set local_judge.base_urls)")
            from openai import AsyncOpenAI  # lazy, as above

            self._clients = [AsyncOpenAI(base_url=u, api_key="EMPTY",
                                         timeout=float(judge_cfg.get("timeout_s", 300)), max_retries=3)
                             for u in urls]
        self._rr = 0

    async def chat(self, messages: list[dict]) -> tuple[str, str | None]:
        client = self._clients[self._rr % len(self._clients)]  # round-robin over replicas
        self._rr += 1
        if self.kind == "local":
            return await _local_chat(client, self.cfg, messages)
        return await _openai_chat(client, self.cfg, messages)

    async def close(self) -> None:
        for c in self._clients:
            await c.close()


def resolve_judge(eval_cfg: dict, judge: str = "gpt54", base_urls=None, concurrency: int | None = None,
                  bench: str | None = None, root: Path = ROOT) -> dict:
    """The judge_cfg of one lane. `gpt54`: the eval config's `judge` block (+ the
    bench's `grader_overrides.gpt54` when `bench` names the OOD benchmark being
    graded). `local`: the grader config named by the eval config's
    `local_judge.grader_config` — model, temperature, extra_body, the recall
    token cap, timeout — plus the replica URLs (the CLI's, else the config's)."""
    if judge not in JUDGES:
        raise SystemExit(f"judge must be {' | '.join(JUDGES)}, got {judge!r}")
    if judge == "gpt54":
        block = dict(eval_cfg.get("judge") or {})
        if not block:
            raise ValueError("the eval config has no `judge` block")
        cfg = {"kind": "openai", **block}
        if bench:
            ood = eval_cfg.get("ood") or {}
            if ood.get("bench") != bench:
                raise ValueError(f"the eval config's OOD bench is {ood.get('bench')!r}, not {bench!r}")
            cfg.update((ood.get("grader_overrides") or {}).get("gpt54") or {})
    else:
        lj = eval_cfg.get("local_judge") or {}
        serve = yaml.safe_load((root / lj.get("grader_config", "configs/grader/qwen36_35b.yaml")).read_text())
        g = serve.get("grader") or {}
        if isinstance(base_urls, str):
            base_urls = base_urls.split(",")
        urls = [u.strip() for u in (base_urls or lj.get("base_urls") or ()) if u and u.strip()]
        cfg = {"kind": "local", "model": (serve.get("serving") or {}).get("model"), "base_urls": urls,
               "temperature": float(g.get("temperature", 0.0)), "extra_body": dict(g.get("extra_body") or {}),
               "max_tokens": int((g.get("max_tokens") or {}).get("coverage", 5000)),
               "timeout_s": float(g.get("timeout_s", 300)), "concurrency": int(lj.get("concurrency", 32))}
    if concurrency:
        cfg["concurrency"] = int(concurrency)
    return cfg


def judge_ident(judge_cfg: dict) -> dict:
    """The manifest's `judge` block: what fixes this judge's output."""
    kind = judge_cfg.get("kind", "openai")
    if kind == "local":
        return {"kind": kind, "model": judge_cfg["model"],
                "base_urls": list(judge_cfg.get("base_urls") or []),
                "temperature": float(judge_cfg.get("temperature", 0.0)),
                "extra_body": dict(judge_cfg.get("extra_body") or {}),
                "max_tokens": int(judge_cfg.get("max_tokens", 5000))}
    ident = {"kind": kind, "model": judge_cfg["model"], "base_url": judge_cfg.get("base_url")}
    if judge_cfg.get("reasoning_effort"):       # recorded only when sent (an OOD grader override)
        ident["reasoning_effort"] = judge_cfg["reasoning_effort"]
    return ident


def check_health(base_urls) -> None:
    """GET each replica's /health; SystemExit on the first one that does not answer 200."""
    import urllib.request

    urls = base_urls.split(",") if isinstance(base_urls, str) else list(base_urls or [])
    for u in (x.strip() for x in urls if x.strip()):
        health = (u[:-len("/v1")] if u.endswith("/v1") else u) + "/health"
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status != 200:
                    raise OSError(f"HTTP {r.status}")
        except Exception as e:  # noqa: BLE001 — any failure means "not serving"
            raise SystemExit(f"grader replica not healthy at {u} ({e}) — start scripts/serve_grader.sh") from None


def add_judge_args(ap) -> None:
    """The judge switch shared by the lane modules."""
    ap.add_argument("--judge", default="gpt54", choices=JUDGES,
                    help="gpt54 = the reference grader (the eval config's judge block, OPENAI_API_KEY); "
                         "local = the training grader on the served replicas")
    ap.add_argument("--base-urls", default=None,
                    help="comma-separated grader replica URLs for --judge local (default: the eval config's)")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="calls in flight (default: the judge's)")
