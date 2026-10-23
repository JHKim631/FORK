"""Shared data structures for prepared prompts and rubric items."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rubric:
    """One rubric criterion with its (signed) weight.

    RubricHub weights are all positive integers 1..20, but the recall
    aggregation supports negative weights (undesirable / pitfall criteria) per
    the general form of the rubric-coverage reward, so weight is a float and
    may be < 0.
    """

    text: str
    weight: float


@dataclass(frozen=True)
class PreparedExample:
    """One training/eval prompt after data preparation (fork.data.prepare)."""

    prompt_id: str
    prompt: str
    rubrics: tuple[Rubric, ...]
    reference_answer: str  # "" for RubricHub
    source: str

    @property
    def rubric_texts(self) -> list[str]:
        return [r.text for r in self.rubrics]
