"""Track optimization attempts and format history for LLM context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Attempt:
    """A single optimization attempt."""

    iteration: int
    strategy: str          # LLM's explanation of what it tried
    score: float | None    # ns/op, None if the attempt failed before measurement
    status: str            # "improved", "regression", "build_failed", "test_failed", "timeout"
    error: str | None      # error message if the attempt failed

    def is_success(self) -> bool:
        return self.status == "improved"


class OptimizationHistory:
    """Accumulates attempts and renders them for inclusion in LLM prompts."""

    def __init__(self) -> None:
        self.attempts: list[Attempt] = []
        self.baseline_score: float = 0.0
        self.best_score: float = float("inf")
        self.best_iteration: int = -1

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, attempt: Attempt) -> None:
        """Record an attempt and update best-score tracking."""
        self.attempts.append(attempt)
        if attempt.score is not None and attempt.score < self.best_score:
            self.best_score = attempt.score
            self.best_iteration = attempt.iteration

    def set_baseline(self, score: float) -> None:
        self.baseline_score = score
        if self.best_score == float("inf"):
            self.best_score = score

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.attempts)

    @property
    def improvements(self) -> int:
        return sum(1 for a in self.attempts if a.status == "improved")

    @property
    def regressions(self) -> int:
        return sum(1 for a in self.attempts if a.status == "regression")

    @property
    def failures(self) -> int:
        return sum(1 for a in self.attempts if a.status not in ("improved", "regression"))

    @property
    def speedup(self) -> float:
        """Best speedup relative to baseline (e.g. 1.5 = 50 % faster)."""
        if self.baseline_score <= 0 or self.best_score == float("inf"):
            return 1.0
        return self.baseline_score / self.best_score

    def failed_strategies(self) -> list[str]:
        """Return strategy descriptions that did NOT improve performance."""
        return [a.strategy for a in self.attempts if a.status != "improved"]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_llm(self, max_recent: int = 5) -> str:
        """Format recent history for inclusion in an LLM prompt.

        Returns a concise, multi-line summary of the last *max_recent*
        attempts so the model can avoid repeating failed strategies and
        build on successful ones.
        """
        if not self.attempts:
            return "No previous attempts."

        recent = self.attempts[-max_recent:]
        lines: list[str] = []

        for a in recent:
            # Truncate long strategy descriptions to keep the prompt lean.
            strategy_brief = a.strategy[:100].replace("\n", " ")

            if a.status == "improved":
                assert a.score is not None
                lines.append(
                    f"Attempt {a.iteration}: {strategy_brief} "
                    f"-> {a.score:.1f} ns/op (IMPROVED)"
                )
            elif a.status == "regression":
                assert a.score is not None
                lines.append(
                    f"Attempt {a.iteration}: {strategy_brief} "
                    f"-> {a.score:.1f} ns/op (SLOWER, rejected)"
                )
            else:
                error_msg = a.error or "unknown"
                lines.append(
                    f"Attempt {a.iteration}: {strategy_brief} "
                    f"-> {a.status.upper()}: {error_msg}"
                )

        header = (
            f"Baseline: {self.baseline_score:.1f} ns/op | "
            f"Best so far: {self.best_score:.1f} ns/op "
            f"({self.speedup:.2f}x speedup)"
        )
        return header + "\n" + "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"OptimizationHistory(attempts={self.total}, "
            f"best={self.best_score:.1f}, baseline={self.baseline_score:.1f})"
        )
