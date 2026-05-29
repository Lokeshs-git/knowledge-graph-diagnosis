"""Scorers — pure functions that grade a model output.

A scorer takes (output, expected) and returns a Score with a value in [0, 1]
plus an optional reason. Compose multiple scorers per eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from quickstart.llm import LLMClient


@dataclass(frozen=True)
class Score:
    """Result of scoring a single example."""

    value: float  # in [0.0, 1.0]
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.value >= 0.5


class Scorer(Protocol):
    """A scorer takes the model's output and the expected value, returns a Score."""

    def __call__(self, output: str, expected: str) -> Score: ...


def exact_match(output: str, expected: str) -> Score:
    """1.0 if output equals expected (case-insensitive, trimmed)."""
    if output.strip().lower() == expected.strip().lower():
        return Score(value=1.0, reason="exact match")
    return Score(value=0.0, reason=f"expected '{expected}', got '{output[:80]}'")


def contains(output: str, expected: str) -> Score:
    """1.0 if expected appears in output (case-insensitive)."""
    if expected.strip().lower() in output.lower():
        return Score(value=1.0, reason=f"contains '{expected}'")
    return Score(value=0.0, reason=f"missing '{expected}'")


class LLMJudge:
    """LLM-as-judge scorer for open-ended outputs.

    Prompts a separate model to grade the output. Useful when exact match
    or substring isn't enough (summaries, explanations, code).

    Tip: use a strong model as the judge even when testing a weaker one.
    """

    DEFAULT_RUBRIC = (
        "Score the model's answer from 0 to 1 based on whether it correctly "
        "addresses the expected answer. Reply with only a JSON object: "
        '{"score": <0.0-1.0>, "reason": "<one sentence>"}'
    )

    def __init__(
        self,
        client: LLMClient | None = None,
        rubric: str | None = None,
    ) -> None:
        self.client = client or LLMClient()
        self.rubric = rubric or self.DEFAULT_RUBRIC

    def __call__(self, output: str, expected: str) -> Score:
        import json

        prompt = (
            f"{self.rubric}\n\n"
            f"Expected answer: {expected}\n"
            f"Model output: {output}\n\n"
            "JSON response:"
        )
        raw = self.client.complete(prompt, max_tokens=200, temperature=0.0)
        try:
            # Strip code fences if present
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
            data = json.loads(cleaned.strip())
            return Score(
                value=float(data.get("score", 0.0)),
                reason=str(data.get("reason", "")),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            return Score(value=0.0, reason=f"judge parse error: {e}")
