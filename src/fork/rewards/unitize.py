"""Unitization: splitting a response into the units precision is measured
over (paper §3.2).

A unit is approximately one sentence of the response, in the response's own
words. The split is deterministic, in four steps:

    A  lines      the response is read line by line; a markdown heading
                  becomes the `section` context of the lines under it, never
                  a unit; list markers and emphasis marks are stripped, the
                  content kept.
    B  sentences  a line splits at `. ? !` followed by whitespace, or at the
                  CJK terminals `。！？`; no split after a bare number or a
                  single letter (decimals, list numbers, initials).
    C  lead-ins   a span ending in a colon announces what follows: it becomes
                  the `intro` context of the units after it, not a unit.
    D  debris     a span with no word character (table rules, bare braces)
                  is not a unit.

Purely functional and organizational text — headings, lead-ins, list markers —
is therefore context rather than content: a unit carries `section` and `intro`,
which the grader prompts render as [section] header lines above the numbered
units (grader/templates.render_units_block). Repeated sentences stay separate
units, so the necessity judgment can see them.

Pure functions, no I/O: `split` is deterministic in the response text, so
units can be recomputed at any time and need no audit-trail persistence.
"""

from __future__ import annotations

import re

__all__ = ["blocks", "sentences", "split"]

_HEADER = re.compile(r"^#{1,6}\s+")
_LIST = re.compile(r"^(?:\d+[.)]|[a-z][.)]|[-*+•])\s+")
_EMPH = re.compile(r"\*\*|__|(?<![a-zA-Z0-9])\*(?![\s*])|(?<![\s*])\*(?![a-zA-Z0-9])")
_CAND = re.compile(r"(?P<ascii>[.?!])\s+(?=\S)|(?P<cjk>[。！？])\s*")
_WORD = re.compile(r"[^\W_]", re.UNICODE)  # Unicode-aware: CJK counts as content


def _clean(s: str) -> str:
    return _EMPH.sub("", s).strip()


def blocks(response: str) -> list[tuple[int, str, str]]:
    """A. -> [(line no, text, section)]. Headers are consumed into `section`
    (a ' > ' path over heading levels) instead of being emitted as blocks."""
    out: list[tuple[int, str, str]] = []
    heads: dict[int, str] = {}
    for n, raw in enumerate(response.split("\n"), start=1):
        s = raw.strip()
        if not s:
            continue
        if _HEADER.match(s):
            level = len(s) - len(s.lstrip("#"))
            heads = {k: v for k, v in heads.items() if k < level}
            heads[level] = _clean(s.lstrip("#")).rstrip(":")
            continue
        text = _clean(_LIST.sub("", s))
        if text:
            out.append((n, text, " > ".join(v for _, v in sorted(heads.items()))))
    return out


def sentences(block: str) -> list[str]:
    """B. Verbatim sentence spans; split guarded against decimals and initials."""
    cuts = []
    for m in _CAND.finditer(block):
        if m.group("ascii"):
            if m.group("ascii") == ".":
                stem = block[: m.start("ascii")]
                tail = re.split(r"[\s(\[]", stem)[-1] if stem else ""
                if tail.isdigit() or len(tail) == 1:
                    continue  # list number / decimal / initial
            cuts.append(m.start("ascii") + 1)
        else:
            cuts.append(m.start("cjk") + 1)
    parts, last = [], 0
    for c in cuts:
        parts.append(block[last:c])
        last = c
    parts.append(block[last:])
    return [p.strip() for p in parts if p.strip()]


def split(response: str) -> list[dict]:
    """The unitizer. -> [{text, line, section, intro}] in response order."""
    units: list[dict] = []
    intro = ""  # nearest colon-terminated announcement (C)
    section = ""
    for line, block, sec in blocks(response):
        if sec != section:
            section, intro = sec, ""
        for sent in sentences(block):
            if sent.endswith(":"):  # C: announcement -> context
                intro = sent.rstrip(":")
            elif _WORD.search(sent):  # D: drop punctuation debris
                units.append({"text": sent, "line": line,
                              "section": section, "intro": intro})
    return units
