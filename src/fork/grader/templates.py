"""Prompt templates: loading prompts/<name>.txt and rendering the blocks the
grader calls fill in.

A template file starts with a header of `#` comment lines (what the prompt is
for and where its text comes from), then a ===SYSTEM=== section and a
===USER=== section. The header is dropped at load time and the two sections
become the chat messages. Rendering uses str.format, so the literal braces a
template contains (the ```json examples) are escaped at load time outside the
known {placeholder} tokens rather than edited in the source files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

_SYS_MARK = "===SYSTEM==="
_USER_MARK = "===USER==="
# Placeholders we substitute; anything else in {curly} is a literal to preserve.
_KNOWN_FIELDS = {
    "prompt",
    "response",
    "rubric_list_string",
    "question",
    "response_A",
    "response_B",
    "reference_block",
    "rubric_block",
    "units_block",
    "judge_range",  # the per-unit calls: the unit numbers this call judges
}


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    user: str

    def render_user(self, **fields: str) -> str:
        return _safe_format(self.user, fields)

    def render_system(self, **fields: str) -> str:
        return _safe_format(self.system, fields)

    def messages(self, **fields: str) -> list[dict[str, str]]:
        """Build an OpenAI-style chat message list. Fields fill BOTH sections."""
        return [
            {"role": "system", "content": _safe_format(self.system, fields)},
            {"role": "user", "content": _safe_format(self.user, fields)},
        ]


def _safe_format(text: str, fields: dict[str, str]) -> str:
    """str.format that only substitutes known placeholders and leaves every other
    brace literal (so ```json examples survive untouched, and a missing field is
    a loud KeyError rather than a silent corruption)."""
    # Escape all braces, then un-escape the known {field} tokens we intend to fill.
    escaped = text.replace("{", "{{").replace("}", "}}")
    for f in _KNOWN_FIELDS:
        escaped = escaped.replace("{{" + f + "}}", "{" + f + "}")
    return escaped.format(**fields)


def _strip_provenance(text: str) -> str:
    lines = text.splitlines()
    kept = []
    in_header = True
    for ln in lines:
        if in_header and ln.lstrip().startswith("#"):
            continue
        in_header = False
        kept.append(ln)
    return "\n".join(kept)


def load_template(name: str) -> PromptTemplate:
    """Load prompts/<name>.txt (e.g. "train/coverage_grading") and split it
    into its system and user sections."""
    path = PROMPTS_DIR / f"{name}.txt"
    raw = path.read_text(encoding="utf-8")
    body = _strip_provenance(raw)
    if _SYS_MARK not in body or _USER_MARK not in body:
        raise ValueError(f"template {name} missing {_SYS_MARK}/{_USER_MARK} markers")
    _, after_sys = body.split(_SYS_MARK, 1)
    system, user = after_sys.split(_USER_MARK, 1)
    return PromptTemplate(name=name, system=system.strip(), user=user.strip())


def render_rubric_list(rubrics: list[str]) -> str:
    """Number rubric criteria 1..k the way the recall grader expects."""
    return "\n".join(f"{i}. {r}" for i, r in enumerate(rubrics, start=1))


def render_units_block(units: list) -> str:
    """Numbered unit list for the per-unit prompts. Unit dicts ({text, section,
    intro}) render grouped under [section / intro] header lines carrying the
    response's own headings and lead-ins — context for the grader, never judged
    (the numbering skips them), so structure carries no reward gradient. Plain
    string units render as a flat list."""
    if units and isinstance(units[0], dict):
        out: list[str] = []
        cur = None
        for i, u in enumerate(units, start=1):
            head = " / ".join(x for x in (u.get("section", ""), u.get("intro", "")) if x)
            if head != cur:
                out.append(f"[{head}]" if head else "[no section]")
                cur = head
            out.append(f"{i}. {u['text']}")
        return "\n".join(out)
    return "\n".join(f"{i}. {u}" for i, u in enumerate(units, start=1))


def render_units_block_scored(units: list[dict], shown: list[str]) -> str:
    """The necessity-threshold call's {units_block}: render_units_block's
    grouping and numbering, EVERY unit tagged with its own credit in the
    two-decimal grain (`(score 0.50)`; rewards/necessity.score_str renders
    `shown`, index i-1 for unit i), zero included — a unit carries its credit
    and nothing else; the call sets the threshold that splits them."""
    if len(shown) != len(units):
        raise ValueError(f"one shown score per unit: {len(shown)} scores for {len(units)} units")
    out: list[str] = []
    cur = None
    for i, u in enumerate(units, start=1):
        head = " / ".join(x for x in (u.get("section", ""), u.get("intro", "")) if x)
        if head != cur:
            out.append(f"[{head}]" if head else "[no section]")
            cur = head
        out.append(f"{i}. (score {shown[i - 1]}) {u['text']}")
    return "\n".join(out)


def render_judge_list(indices) -> str:
    """The per-unit calls' {judge_range}: a compact, sorted list of the
    assigned unit numbers — runs of three or more collapse to a range, pairs
    stay explicit: [2, 3, 5, 8, 9, 10, 11] -> "2, 3, 5, 8-11"."""
    idx = sorted({int(i) for i in indices})
    parts: list[str] = []
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        if j - i >= 2:
            parts.append(f"{idx[i]}-{idx[j]}")
        else:
            parts.extend(str(idx[k]) for k in range(i, j + 1))
        i = j + 1
    return ", ".join(parts)


def render_reference_block(reference: str | None) -> str:
    """RubricHub prompts carry no reference answer, so this is usually empty."""
    if not reference or not reference.strip():
        return ""
    return f"\n<reference>\n{reference.strip()}\n</reference>\n"
