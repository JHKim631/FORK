"""Async client for the training grader — a self-hosted LLM served by vLLM
behind OpenAI-compatible endpoints.

The client owns the transport and the five grader calls the rewards make:

    grade_recall      the rubric-coverage grading (the recall term; every reward)
    label             the Generic Criteria labeler (correct ∧ necessary per unit), chunked
    match_rubrics     FORK step 1: which units are the basis of each MET rubric
    threshold         FORK step 2: the necessity threshold + the keep / remove verdicts
    correctness       FORK step 3: the factual-correctness verdicts of the necessary units
    precision_credit  the FORK chain: match -> credits -> threshold -> correctness -> precision

Transport: round-robin over the replicas, each guarded by an in-flight
semaphore; retry with exponential backoff on transport / 5xx / timeout; the
truncation policy of `_run` (a body cut at max_tokens is retried once at 1.5×
the budget and never parsed otherwise). Every call returns a `CallResult` with
the parsed value AND the raw model text + token counts, so the caller can both
compute the reward and write the exact IO to the audit trail. Parse, transport
and truncation failures never raise out of a call method — they come back as
`ok=False` with `error` set, which the reward's fallback policy keys on
(rewards/fallback.py).

Loop affinity: the semaphores and the HTTP pools inside AsyncOpenAI bind to
the first event loop that uses them. Reusing a client across asyncio.run()
calls would surface as transport errors on every call — which the reward
fallback would quietly turn into zero rewards for whole batches — so calls
from a second loop fail LOUDLY instead. Construct a fresh client inside each
asyncio.run() context; construction is cheap.

Optional on-disk cache: pass cache_path to memoize successful calls across
process restarts, keyed by a hash of the full rendered request. Failures are
never cached. Meant for evaluation re-runs; training rollouts are freshly
sampled every step, so the trainer leaves it off.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..data.schemas import Rubric
from ..rewards.credit import (
    matching_counters,
    parse_coverage_verdicts,
    parse_matching,
    render_verdict_feed,
    unit_credits,
)
from ..rewards.generic import CRITERIA, generic_precision, parse_unit_labels_salvage
from ..rewards.necessity import chunk_by_scored, merge_threshold_parts, parse_threshold_call, score_str
from ..rewards.precision import MAX_UNITS, chunk_ranges, compose_precision, select_label_sample
from ..rewards.recall import recall_from_text
from ..rewards.verdicts import (
    CHUNK,
    CORRECTNESS_VOCAB,
    NECESSITY_VOCAB,
    chunk_index_list,
    parse_verdicts_listed,
)
from .parsing import ParseError, loads_lenient
from .templates import (
    load_template,
    render_judge_list,
    render_reference_block,
    render_rubric_list,
    render_units_block,
    render_units_block_scored,
)

__all__ = ["GraderConfig", "CallResult", "GraderClient", "DOMAINS", "is_refusal"]

# The wording of the prompts that address a field: `medical` loads the prompt
# files the paper prints, `science` their `_science` twins. Selected by the run
# config's `domain`, never inferred from the data.
DOMAINS = frozenset({"medical", "science"})

# Transport-level failures worth retrying. 4xx (bad request) is NOT here: retrying
# a malformed request just wastes the budget — that surfaces as a failed call.
_RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


@dataclass(frozen=True)
class GraderConfig:
    """The grader's call parameters (configs/grader/<model>.yaml, `grader:` block)."""

    model: str
    temperature: float
    max_tokens_coverage: int
    max_tokens_label: int
    max_tokens_match: int
    max_tokens_threshold: int
    max_tokens_correctness: int
    concurrency_per_replica: int
    timeout_s: float
    max_retries: int
    # Extra per-request body merged into every chat call — the grader's own
    # reasoning control (Qwen3 silences its think channel via
    # {"chat_template_kwargs": {"enable_thinking": false}}). Empty for a plain model.
    extra_body: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict) -> GraderConfig:
        """Build from a parsed configs/grader/<model>.yaml dict."""
        serving = cfg.get("serving", {})
        g = cfg.get("grader", {})
        mt = g.get("max_tokens", {})
        return cls(
            model=serving.get("model", "Qwen/Qwen3.6-35B-A3B-FP8"),
            temperature=float(g.get("temperature", 0.0)),
            max_tokens_coverage=int(mt.get("coverage", 5000)),
            max_tokens_label=int(mt.get("label", 5000)),
            max_tokens_match=int(mt.get("match", 4000)),
            max_tokens_threshold=int(mt.get("threshold", 8000)),
            max_tokens_correctness=int(mt.get("correctness", 8000)),
            concurrency_per_replica=int(g.get("concurrency_per_replica", 64)),
            timeout_s=float(g.get("timeout_s", 180)),
            max_retries=int(g.get("max_retries", 2)),
            extra_body=dict(g.get("extra_body") or {}),
        )


@dataclass
class CallResult:
    """Outcome of one grader call.

    `value` depends on `call_type`:
      coverage    -> RecallResult      (rewards/recall.py)
      label       -> PrecisionResult   (rewards/generic.py)
      match       -> {"entries", "extra_entries", "met", "n_verdicts"}  (rewards/credit.py)
      threshold   -> (entries, flags, threshold, necessary)             (rewards/necessity.py)
      correctness -> (entries, flags)                                    (rewards/verdicts.py)
      precision   -> PrecisionResult   (the FORK chain's composed result)
    `value` is None when `ok` is False. `raw` is "" when the HTTP call itself
    failed (transport), non-empty (but unparseable) on a parse failure.

    `finish_reason` is the model's stop reason ("stop", "length", ...).
    Truncation policy: "length" counts as a failed attempt. _run retries ONCE
    at 1.5x max_tokens (temperature-0 decoding just continues further and
    completes the JSON); if the retry is still truncated the call fails
    (`ok=False`) and the body is NEVER parsed — json-repair would otherwise
    salvage a silently deflated (or example-echoing) payload. So `ok=True`
    implies the final attempt finished with a real stop. `truncation_retries`
    counts budget escalations. `prompt_tokens` / `completion_tokens` /
    `latency_s` accumulate over attempts. `cached` marks a value served from
    the on-disk cache (latency 0, no tokens spent this run).
    """

    call_type: str
    ok: bool
    value: object
    raw: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None
    finish_reason: str | None = None
    truncation_retries: int = 0
    refusal_retries: int = 0  # label only: 1 = the audit-template rescue fired (see label())
    cached: bool = False

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


# Refusal signature for the label rescue path: a short non-JSON body apologizing
# instead of grading ("I'm sorry, but I can't comply with that."). Deliberately
# conservative — only ever consulted AFTER the primary call already failed to
# parse, so a false negative just means no rescue attempt, never a lost grade.
_REFUSAL_RE = re.compile(
    r"(i[’']?m sorry|i can[’']?t|i cannot|can[’']?t (?:comply|provide|complete|assist|help))",
    re.IGNORECASE,
)


def is_refusal(raw: str) -> bool:
    text = raw.strip()
    return 0 < len(text) < 400 and bool(_REFUSAL_RE.search(text))


class GraderClient:
    def __init__(
        self,
        base_urls: list[str],
        cfg: GraderConfig,
        api_key: str = "EMPTY",
        cache_path: str | Path | None = None,
        domain: str = "medical",
        label_templates: tuple[str, str] = ("train/generic_criteria", "train/generic_criteria_audit"),
    ):
        if not base_urls:
            raise ValueError("need at least one base_url")
        if domain not in DOMAINS:
            raise ValueError(f"domain must be {'|'.join(sorted(DOMAINS))}, got {domain!r}")
        self.cfg = cfg
        self.domain = domain
        _dom = "" if domain == "medical" else f"_{domain}"
        # One AsyncOpenAI + semaphore per replica. We do our own backoff, so the
        # SDK's built-in retries are disabled (max_retries=0).
        self._clients = [
            AsyncOpenAI(base_url=u, api_key=api_key, timeout=cfg.timeout_s, max_retries=0)
            for u in base_urls
        ]
        self._sems = [asyncio.Semaphore(cfg.concurrency_per_replica) for _ in base_urls]
        self._rr = 0
        self._loop: asyncio.AbstractEventLoop | None = None  # loop-affinity guard; set on first call
        # Templates loaded once; rendering is pure and cheap thereafter. The
        # recall prompt is domain-neutral; the three FORK prompts and the
        # label pair address a field.
        self._t_cov = load_template("train/coverage_grading")
        # The Generic Criteria labeler + its refusal-rescue twin (label()). The
        # evaluation's precision grader passes its own pair (eval/precision_label
        # and its audit twin); that primary has a science twin, the training one
        # is medical-only.
        primary, audit = label_templates
        if domain != "medical" and primary.startswith("eval/"):
            primary = f"{primary}{_dom}"
        self._t_label = load_template(primary)
        self._t_label_audit = load_template(audit)
        self._t_match = load_template(f"train/fork_match_rubrics{_dom}")
        self._t_thr = load_template(f"train/fork_necessity_threshold{_dom}")
        self._t_cor = load_template(f"train/fork_correctness{_dom}")
        # Optional on-disk cache (module docstring). Torn lines from a concurrent
        # writer are skipped on load and simply re-verified live.
        self._cache: dict[str, dict] | None = None
        self._cache_f = None
        if cache_path is not None:
            p = Path(cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._cache = {}
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    try:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
            self._cache_f = p.open("a", encoding="utf-8")

    # --- transport ---------------------------------------------------------------------

    def _pick(self) -> int:
        i = self._rr
        self._rr = (self._rr + 1) % len(self._clients)
        return i

    def _bind_loop(self) -> None:
        """Fail loudly on cross-loop reuse instead of degrading into transport
        errors that the fallback policy would silently turn into zero rewards."""
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif loop is not self._loop:
            raise RuntimeError(
                "GraderClient reused across event loops: its semaphores/HTTP pools "
                "are bound to a previous asyncio.run(); construct a new client inside "
                "the current loop (see module docstring)"
            )

    def _cache_key(self, call_type: str, messages: list[dict]) -> str | None:
        """Hash of everything that determines the grader's output. max_tokens is
        deliberately excluded: a cached body always came from a completed
        (finish != length) call, so it stays valid when the caps are retuned."""
        if self._cache is None:
            return None
        payload = json.dumps(
            [self.cfg.model, self.cfg.temperature, self.cfg.extra_body, call_type, messages],
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def _chat(self, messages: list[dict], max_tokens: int, temperature: float | None = None):
        """One chat/completions call to the next replica.

        Returns (content, prompt_tok, completion_tok, finish_reason). finish_reason
        == "length" means the output hit max_tokens (truncated) — surfaced so the
        caller can detect the silent score deflation that follows truncation.

        Retries transport errors with exponential backoff; a persistent failure
        re-raises (the caller's _run turns it into a failed CallResult).
        """
        idx = self._pick()
        client, sem = self._clients[idx], self._sems[idx]
        extra = dict(self.cfg.extra_body) or None
        temp = self.cfg.temperature if temperature is None else temperature
        async with sem:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.cfg.max_retries + 1),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                retry=retry_if_exception_type(_RETRYABLE),
                reraise=True,
            ):
                with attempt:
                    resp = await client.chat.completions.create(
                        model=self.cfg.model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_tokens,
                        extra_body=extra,
                    )
                    choice = resp.choices[0]
                    usage = resp.usage
                    return (
                        choice.message.content or "",
                        usage.prompt_tokens if usage else 0,
                        usage.completion_tokens if usage else 0,
                        choice.finish_reason,
                    )
        raise RuntimeError("unreachable: AsyncRetrying reraises on exhaustion")  # pragma: no cover

    async def _run(self, call_type, messages, max_tokens, parse_fn) -> CallResult:
        """One grader call with the truncation policy: finish_reason == "length"
        is a failed attempt — retry ONCE at 1.5x max_tokens (at temperature 0
        the decode continues past the old ceiling and completes the JSON).
        Still truncated after the retry -> the call fails WITHOUT parsing: a
        truncated body must never be salvaged into a deflated score or an
        echoed-example payload."""
        self._bind_loop()
        key = self._cache_key(call_type, messages)
        if key is not None and (hit := self._cache.get(key)) is not None:
            try:
                value = parse_fn(hit["raw"])
            except Exception:
                pass  # corrupt/stale entry: fall through to a live call (and overwrite)
            else:
                return CallResult(call_type, True, value, hit["raw"], 0.0,
                                  int(hit.get("prompt_tokens", 0)),
                                  int(hit.get("completion_tokens", 0)),
                                  None, hit.get("finish_reason"), 0, cached=True)
        t0 = time.perf_counter()
        tokens = max_tokens
        pt_total = ct_total = 0
        retries = 0
        while True:
            try:
                raw, pt, ct, fr = await self._chat(messages, tokens)
            except Exception as e:  # transport failure after retries
                return CallResult(call_type, False, None, "", time.perf_counter() - t0,
                                  pt_total, ct_total, f"{type(e).__name__}: {e}",
                                  truncation_retries=retries)
            pt_total += pt
            ct_total += ct
            if fr != "length":
                break
            if retries >= 1:
                return CallResult(
                    call_type, False, None, raw, time.perf_counter() - t0, pt_total, ct_total,
                    f"truncated: finish_reason=length at max_tokens={tokens} even after 1.5x retry",
                    finish_reason=fr, truncation_retries=retries)
            retries += 1
            tokens = int(tokens * 1.5)
        try:
            value = parse_fn(raw)
        except ParseError as e:
            return CallResult(call_type, False, None, raw, time.perf_counter() - t0,
                              pt_total, ct_total, f"ParseError: {e}",
                              finish_reason=fr, truncation_retries=retries)
        except Exception as e:  # defensive: unexpected shape from parse_fn
            return CallResult(call_type, False, None, raw, time.perf_counter() - t0,
                              pt_total, ct_total, f"{type(e).__name__}: {e}",
                              finish_reason=fr, truncation_retries=retries)
        if key is not None:  # cache only calls that completed AND parsed
            rec = {"key": key, "call_type": call_type, "raw": raw, "finish_reason": fr,
                   "prompt_tokens": pt_total, "completion_tokens": ct_total}
            self._cache[key] = rec
            self._cache_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._cache_f.flush()
        return CallResult(call_type, True, value, raw, time.perf_counter() - t0,
                          pt_total, ct_total, None,
                          finish_reason=fr, truncation_retries=retries)

    # --- recall ------------------------------------------------------------------------

    async def grade_recall(
        self, rubrics: list[Rubric], prompt: str, response: str
    ) -> CallResult:
        """Rubric recall: one grading call over all rubrics. value=RecallResult."""
        rubrics = list(rubrics)
        msgs = self._t_cov.messages(
            prompt=prompt,
            response=response,
            rubric_list_string=render_rubric_list([r.text for r in rubrics]),
        )
        return await self._run(
            "coverage", msgs, self.cfg.max_tokens_coverage,
            lambda raw: recall_from_text(rubrics, raw),
        )

    # --- Generic Criteria --------------------------------------------------------------

    async def label(
        self, question: str, units: list[dict], rubrics: list[Rubric], reference: str = "",
    ) -> CallResult:
        """The Generic Criteria labeler: correct ∧ necessary per unit, every
        unit labeled in chunks of LABEL_CHUNK. value=PrecisionResult.

        All units are rendered in every call (grouped under [section] headers)
        with {judge_range} assigning the judged slice — this keeps cross-chunk
        redundancy visible to the labeler. Chunks run concurrently; each gets
        its own refusal rescue: when the primary call fails with a
        refusal-shaped ParseError, the SAME payload is retried ONCE with the
        audit-framed twin template (`refusal_retries=1` marks the result).

        No units (an empty response) short-circuits to precision 0 with no
        call. Above MAX_UNITS a seeded uniform subsample is labeled (the true
        unit count is still reported). ANY chunk failing fails the whole call —
        partial credit would bias precision by length — so the group-median
        fallback applies unchanged. The merged CallResult: `value` = the
        composition over all labeled units; `raw` = one JSON object of the
        model's per-unit entries keyed by global index; tokens and retries sum
        over chunks; `latency_s` is the slowest chunk; `cached` only if every
        chunk was served from cache.
        """
        if not units:
            return CallResult("label", True, generic_precision([], {}), "", 0.0, 0, 0)
        sample = select_label_sample(units, cap=MAX_UNITS)
        fields = dict(
            question=question,
            reference_block=render_reference_block(reference),
            rubric_block=render_rubric_list([r.text for r in rubrics]),
            units_block=render_units_block(sample),
        )

        async def one(lo: int, hi: int) -> CallResult:
            f = dict(fields, judge_range=f"{lo}-{hi}")
            parse = lambda raw: parse_unit_labels_salvage(raw, hi - lo + 1, first=lo, keys=CRITERIA)  # noqa: E731
            res = await self._run(
                "label", self._t_label.messages(**f), self.cfg.max_tokens_label, parse
            )
            if res.ok or not (res.error or "").startswith("ParseError") or not is_refusal(res.raw):
                return res
            rescue = await self._run(
                "label", self._t_label_audit.messages(**f), self.cfg.max_tokens_label, parse
            )
            return replace(rescue, refusal_retries=rescue.refusal_retries + 1)

        ranges = chunk_ranges(len(sample))
        chunks = await asyncio.gather(*(one(lo, hi) for lo, hi in ranges))

        latency = max(c.latency_s for c in chunks)
        pt = sum(c.prompt_tokens for c in chunks)
        ct = sum(c.completion_tokens for c in chunks)
        tr = sum(c.truncation_retries for c in chunks)
        rr = sum(c.refusal_retries for c in chunks)
        failed = [(r, c) for r, c in zip(ranges, chunks) if not c.ok]
        if failed:
            (lo, hi), first = failed[0]
            more = f" (+{len(failed) - 1} more failed chunks)" if len(failed) > 1 else ""
            return CallResult(
                "label", False, None, first.raw, latency, pt, ct,
                f"chunk {lo}-{hi} of m={len(sample)}: {first.error}{more}",
                finish_reason=first.finish_reason, truncation_retries=tr, refusal_retries=rr)

        labels: dict[int, dict[str, bool]] = {}
        entries: dict[str, object] = {}  # the model's own per-unit JSON entries
        for (lo, hi), c in zip(ranges, chunks):
            labels.update(c.value)
            try:
                obj = loads_lenient(c.raw)
            except Exception:
                obj = {}
            if isinstance(obj, dict):
                for i in range(lo, hi + 1):
                    if str(i) in obj:
                        entries[str(i)] = obj[str(i)]
        value = generic_precision(sample, labels, total_units=len(units))
        return CallResult(
            "label", True, value, json.dumps(entries, ensure_ascii=False), latency,
            pt, ct, None, finish_reason="stop", truncation_retries=tr,
            refusal_retries=rr, cached=all(c.cached for c in chunks))

    # --- FORK: credit-based necessity + factual correctness ---------------------------

    async def match_rubrics(self, question: str, units: list[dict], rubrics: list[Rubric],
                            cov_raw: str) -> CallResult:
        """FORK step 1: which units are the basis of each MET rubric. value =
        {"entries": {j: entry}, "extra_entries", "met", "n_verdicts"} — entries in
        the shape rewards.credit.credit_sets consumes. The feed is met-only,
        rendered from the recall grader's own raw JSON. No met rubric, or no
        units -> a synthesized ok result with no entries and no call (every
        unit then carries no credit)."""
        texts = [r.text for r in rubrics]
        verdicts = parse_coverage_verdicts(cov_raw, len(texts))
        met = sorted(j for j, (m, _why) in verdicts.items() if m)
        base = {"met": met, "n_verdicts": len(verdicts)}
        if not units or not met:
            return CallResult("match", True, {"entries": {}, "extra_entries": 0, **base},
                              "", 0.0, 0, 0, finish_reason="stop")
        msgs = self._t_match.messages(
            question=question,
            rubric_block=render_verdict_feed(texts, verdicts),
            units_block=render_units_block(units),
        )
        n = len(units)
        parse = lambda raw: {**parse_matching(raw, met, n), **base}  # noqa: E731
        return await self._run("match", msgs, self.cfg.max_tokens_match, parse)

    async def threshold(self, question: str, units: list[dict], shown: list[str]) -> CallResult:
        """FORK step 2: the necessity threshold call. Every unit is rendered with
        its credit in every call (render_units_block_scored); the calls cover
        runs of consecutive units holding at most CHUNK units with a positive
        credit (chunk_by_scored); each call returns its threshold for the whole
        response and the verdicts for the units of its run at or above it
        (parse_threshold_call). Bail = fewer than half of those units valid; a
        failed or bailed call is retried ONCE as two half-runs, each half a
        part of its own; ANY run failing fails the call. The parts merge by
        merge_threshold_parts (the strictest threshold wins, each unit's
        verdict from its own part). value = (entries, flags, threshold,
        necessary): the merged entries over the units at or above the
        threshold, the summed counters (+ n_calls, thr_disagree, thr_fallback,
        thr_fallback_calls, thr_dropped_by_merge, thr_cites_below), the
        response threshold (a Fraction in the credit grain) and the list of
        units at or above it. raw = the merged JSON with "threshold" first."""
        tpl, vocab, cap = self._t_thr, NECESSITY_VOCAB, self.cfg.max_tokens_threshold
        fields = dict(question=question, units_block=render_units_block_scored(units, shown))

        async def call(run: list[int]) -> tuple[CallResult, bool]:
            parse = lambda raw: parse_threshold_call(raw, run, shown, vocab)  # noqa: E731
            res = await self._run("threshold", tpl.messages(**fields, judge_range=render_judge_list(run)),
                                  cap, parse)
            bail = res.ok and len(res.value[1]) < res.value[2]["n_necessary_run"] / 2
            return res, bail

        async def one_chunk(run: list[int]) -> dict:
            res, bail = await call(run)
            ch = {"lo": run[0], "hi": run[-1], "n_assigned": len(run), "latency": res.latency_s,
                  "pt": res.prompt_tokens, "ct": res.completion_tokens, "tr": res.truncation_retries,
                  "cached": res.cached, "split_retry": False, "split_recovered": False}
            if res.ok and not bail:
                return {**ch, "ok": True, "parts": [res.value]}
            if len(run) > 1:  # SPLIT RETRY (one level; a single-unit run cannot split)
                ch["split_retry"] = True
                mid = len(run) // 2
                subs = await asyncio.gather(call(run[:mid]), call(run[mid:]))
                ch["latency"] += max(r.latency_s for r, _b in subs)
                ch["pt"] += sum(r.prompt_tokens for r, _b in subs)
                ch["ct"] += sum(r.completion_tokens for r, _b in subs)
                ch["tr"] += sum(r.truncation_retries for r, _b in subs)
                ch["cached"] = ch["cached"] and all(r.cached for r, _b in subs)
                if all(r.ok and not b for r, b in subs):
                    ch["split_recovered"] = True
                    return {**ch, "ok": True, "parts": [r.value for r, _b in subs]}
            err = (f"bail: {len(res.value[1])} valid of {res.value[2]['n_necessary_run']} units above the threshold"
                   if res.ok else res.error)
            if ch["split_retry"]:
                err = f"{err} (after split retry)"
            return {**ch, "ok": False, "error": err, "raw": res.raw,
                    "finish_reason": res.finish_reason, "parts": []}

        chunks = await asyncio.gather(*(one_chunk(r) for r in chunk_by_scored(shown, CHUNK)))
        latency = max(c["latency"] for c in chunks)
        pt = sum(c["pt"] for c in chunks)
        ct = sum(c["ct"] for c in chunks)
        tr = sum(c["tr"] for c in chunks)
        failed = [c for c in chunks if not c["ok"]]
        if failed:
            first = failed[0]
            more = f" (+{len(failed) - 1} more failed chunks)" if len(failed) > 1 else ""
            return CallResult(
                "threshold", False, None, first["raw"], latency, pt, ct,
                f"chunk {first['lo']}-{first['hi']} ({first['n_assigned']} units) of "
                f"N={len(units)}: {first['error']}{more}",
                finish_reason=first.get("finish_reason"), truncation_retries=tr)
        thr, entries, necessary, flags = merge_threshold_parts([p for c in chunks for p in c["parts"]], shown)
        flags["split_retries"] = sum(c["split_retry"] for c in chunks)
        flags["split_recovered"] = sum(c["split_recovered"] for c in chunks)
        flags["n_chunks"] = len(chunks)
        raw = json.dumps({"threshold": score_str(thr),
                          **{str(i): entries[i] for i in sorted(entries)}}, ensure_ascii=False)
        return CallResult("threshold", True, (entries, flags, thr, necessary), raw, latency, pt, ct, None,
                          finish_reason="stop", truncation_retries=tr,
                          cached=all(c["cached"] for c in chunks))

    async def correctness(self, question: str, units: list[dict], assigned: list[int]) -> CallResult:
        """FORK step 3: correct | incorrect | uncertain | na over the ASSIGNED
        (necessary) units only, every unit rendered untagged in every call.
        Chunks of CHUNK over the assigned list; bail = fewer than half a
        chunk's assigned units valid; a failed or bailed chunk is retried ONCE
        as the two halves of its list; ANY chunk failing fails the whole call.
        value = (entries, flags); raw = the merged JSON of the entries."""
        tpl, vocab, cap = self._t_cor, CORRECTNESS_VOCAB, self.cfg.max_tokens_correctness
        fields = dict(question=question, units_block=render_units_block(units))

        async def call(idx: list[int]) -> tuple[CallResult, bool]:
            parse = lambda raw: parse_verdicts_listed(raw, idx, vocab)  # noqa: E731
            res = await self._run("correctness", tpl.messages(**fields, judge_range=render_judge_list(idx)),
                                  cap, parse)
            bail = res.ok and len(res.value[0]) < len(idx) / 2
            return res, bail

        async def one_chunk(idx: list[int]) -> dict:
            res, bail = await call(idx)
            ch = {"lo": idx[0], "hi": idx[-1], "n_assigned": len(idx), "latency": res.latency_s,
                  "pt": res.prompt_tokens, "ct": res.completion_tokens, "tr": res.truncation_retries,
                  "cached": res.cached, "split_retry": False, "split_recovered": False}
            if res.ok and not bail:
                entries, flags = res.value
                return {**ch, "ok": True, "entries": entries, "flags": Counter(flags)}
            if len(idx) > 1:  # SPLIT RETRY (one level; a single-unit chunk cannot split)
                ch["split_retry"] = True
                mid = len(idx) // 2
                subs = await asyncio.gather(call(idx[:mid]), call(idx[mid:]))
                ch["latency"] += max(r.latency_s for r, _b in subs)
                ch["pt"] += sum(r.prompt_tokens for r, _b in subs)
                ch["ct"] += sum(r.completion_tokens for r, _b in subs)
                ch["tr"] += sum(r.truncation_retries for r, _b in subs)
                ch["cached"] = ch["cached"] and all(r.cached for r, _b in subs)
                if all(r.ok and not b for r, b in subs):
                    entries: dict[int, dict] = {}
                    flags: Counter = Counter()
                    for r, _b in subs:
                        e, f = r.value
                        entries.update(e)
                        flags.update(f)
                    ch["split_recovered"] = True
                    return {**ch, "ok": True, "entries": entries, "flags": flags}
            err = (f"bail: {len(res.value[0])} valid of {len(idx)} assigned units"
                   if res.ok else res.error)
            if ch["split_retry"]:
                err = f"{err} (after split retry)"
            return {**ch, "ok": False, "error": err, "raw": res.raw,
                    "finish_reason": res.finish_reason, "entries": {}, "flags": Counter()}

        chunks = await asyncio.gather(*(one_chunk(c) for c in chunk_index_list(assigned, CHUNK)))
        latency = max(c["latency"] for c in chunks)
        pt = sum(c["pt"] for c in chunks)
        ct = sum(c["ct"] for c in chunks)
        tr = sum(c["tr"] for c in chunks)
        failed = [c for c in chunks if not c["ok"]]
        if failed:
            first = failed[0]
            more = f" (+{len(failed) - 1} more failed chunks)" if len(failed) > 1 else ""
            return CallResult(
                "correctness", False, None, first["raw"], latency, pt, ct,
                f"chunk {first['lo']}-{first['hi']} ({first['n_assigned']} assigned) of "
                f"N={len(units)}: {first['error']}{more}",
                finish_reason=first.get("finish_reason"), truncation_retries=tr)
        entries = {}
        flags: Counter = Counter()
        for c in chunks:
            entries.update(c["entries"])
            flags.update(c["flags"])
        flags["split_retries"] = sum(c["split_retry"] for c in chunks)
        flags["split_recovered"] = sum(c["split_recovered"] for c in chunks)
        flags["n_chunks"] = len(chunks)
        raw = json.dumps({str(i): entries[i] for i in sorted(entries)}, ensure_ascii=False)
        return CallResult("correctness", True, (entries, dict(flags)), raw, latency, pt, ct, None,
                          finish_reason="stop", truncation_retries=tr,
                          cached=all(c["cached"] for c in chunks))

    async def precision_credit(self, question: str, units: list[dict], rubrics: list[Rubric],
                               cov: CallResult) -> dict:
        """The FORK precision chain, one implementation for training and
        evaluation: recall verdicts -> match_rubrics -> credits -> threshold
        (over every unit, each shown with its credit) -> correctness over the
        units at or above the threshold -> compose_precision. Returns
        {"match", "threshold", "correctness", "precision"}: the three calls and
        the composed result (a `precision` CallResult whose value is the
        PrecisionResult; its telemetry carries the chain's reads).

        Two counted short circuits: no unit with a positive credit (no
        threshold call, precision 0) and a threshold above every credit
        (correctness skipped, precision 0) — `thr_stage` in the telemetry says
        which (`no_scored_unit` | `none_above` | `judged`). Zero units ->
        precision 0 with no call. Every stage indexes ONE unit list: the seeded
        MAX_UNITS subsample, so unit numbers agree across the three calls.
        A failed stage fails the chain (`ok=False`, the stage named), and the
        group-median fallback applies."""
        out: dict = {"match": None, "threshold": None, "correctness": None, "precision": None}

        def failed(stage: str, err: str | None, raw: str = "") -> CallResult:
            return CallResult("precision", False, None, raw, 0.0, 0, 0,
                              f"credit chain failed at {stage}: {err}")

        if not units:
            out["precision"] = CallResult("precision", True, compose_precision(0, set(), {}, {}), "",
                                          0.0, 0, 0, finish_reason="stop")
            return out
        if not cov.ok:
            out["precision"] = failed("recall", cov.error)
            return out
        sample = select_label_sample(units, cap=MAX_UNITS)
        n, m = len(sample), len(units)
        match = await self.match_rubrics(question, sample, rubrics, cov.raw)
        out["match"] = match
        if not match.ok:
            out["precision"] = failed("match", match.error, match.raw)
            return out
        credits = unit_credits(match.value["entries"], n)
        shown = [score_str(u.credit) for u in credits]
        n_scored = sum(1 for s in shown if s != "0.00")
        tele = {
            "n_units": m, "n_labeled": n,
            "n_met": len(match.value["met"]), "n_verdicts": match.value["n_verdicts"],
            "match": matching_counters(match.value),
            "mean_credit": (sum(float(u.credit) for u in credits) / n) if n else None,
            "floor": float(credits[0].floor) if credits else None,
            "credits": shown, "n_scored": n_scored,
            "threshold": None, "threshold_fallback": False, "n_calls": 0,
            "thr_disagree": False, "thr_cites_below": 0,
        }
        if n_scored == 0:   # short circuit 1: no unit can clear a positive threshold — no call
            tele.update({"thr_stage": "no_scored_unit", "necessary": [], "n_necessary": 0,
                         "necessity_flags": {}, "correctness_flags": {}})
            value = compose_precision(n, set(), {}, {}, units=sample, total_units=m, telemetry=tele)
            out["precision"] = CallResult("precision", True, value, "", 0.0, 0, 0, None, finish_reason="stop")
            return out
        thr = await self.threshold(question, sample, shown)
        out["threshold"] = thr
        if not thr.ok:
            out["precision"] = failed("threshold", thr.error, thr.raw)
            return out
        thr_entries, thr_flags, tau, necessary = thr.value
        tele.update({
            "threshold": score_str(tau), "threshold_fallback": bool(thr_flags.get("thr_fallback")),
            "n_calls": int(thr_flags.get("n_calls") or 0), "thr_disagree": bool(thr_flags.get("thr_disagree")),
            "thr_cites_below": int(thr_flags.get("thr_cites_below") or 0),
            "necessary": necessary, "n_necessary": len(necessary),
        })
        if not necessary:   # short circuit 2: the threshold cleared every credit — correctness skipped
            tele.update({"thr_stage": "none_above", "necessity_flags": thr_flags, "correctness_flags": {}})
            value = compose_precision(n, set(), {}, {}, units=sample, total_units=m, telemetry=tele)
            out["precision"] = CallResult("precision", True, value, "", thr.latency_s, thr.prompt_tokens,
                                          thr.completion_tokens, None, finish_reason="stop",
                                          truncation_retries=thr.truncation_retries, cached=thr.cached)
            return out
        cor = await self.correctness(question, sample, necessary)
        out["correctness"] = cor
        if not cor.ok:
            out["precision"] = failed("correctness", cor.error, cor.raw)
            return out
        cor_entries, cor_flags = cor.value
        tele.update({"thr_stage": "judged", "necessity_flags": thr_flags, "correctness_flags": cor_flags})
        value = compose_precision(n, set(necessary), thr_entries, cor_entries, units=sample,
                                  total_units=m, telemetry=tele)
        out["precision"] = CallResult(
            "precision", True, value, "", thr.latency_s + cor.latency_s,
            thr.prompt_tokens + cor.prompt_tokens, thr.completion_tokens + cor.completion_tokens,
            None, finish_reason="stop", truncation_retries=thr.truncation_retries + cor.truncation_retries,
            cached=thr.cached and cor.cached)
        return out

    async def aclose(self) -> None:
        if self._cache_f is not None:
            self._cache_f.close()
            self._cache_f = None
        for c in self._clients:
            await c.close()
