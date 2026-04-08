#!/usr/bin/env python3
"""
Plot AutoPerf Exploration vs Exploitation Dynamics (2-panel: phases + volatility).

Parses AutoPerf stdout log format:
    [d1/b0] #1  IMPROVED  3.8 ns/op  (best: 5.3)
    [d1/b1] #2  REGRESSION  4.5 ns/op  (best: 3.8)
    [d1/b2] #3  BUILD FAILED  N/A

Usage:
    python scripts/plot_exploration_dynamics.py \
        --log matmul_log.txt --baseline 126460 \
        --output plots/exploration.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np

# ── Colours ──────────────────────────────────────────────────────────
GREEN = "#22c55e"
RED = "#ef4444"
PURPLE = "#6d28d9"
CORAL = "#D97757"
GRAY = "#9ca3af"

PHASE_COLORS = {
    "Exploration": "#dbeafe",   # light blue
    "Discovery": "#dcfce7",     # light green
    "Refinement": "#fef3c7",    # light amber
}
PHASE_LABEL_COLORS = {
    "Exploration": "#3b82f6",
    "Discovery": "#22c55e",
    "Refinement": "#f59e0b",
}

# ── Data structures ──────────────────────────────────────────────────

class Iteration(NamedTuple):
    eval_num: int
    status: str
    score: float | None
    best_so_far: float | None


# ── Log parser (shared logic) ────────────────────────────────────────

_LINE_RE = re.compile(
    r"#(\d+)\s+"
    r"(IMPROVED|REGRESSION|BUILD FAILED|TEST FAILED|TIMEOUT|NO_CODE|BENCH FAILED|"
    r"EXTRACTION_FAILED|LLM_ERROR|NO CODE)\s+"
    r"([\d.eE+\-]+)\s*ns/op\s+\(best:\s*([\d.eE+\-]+)\)",
    re.IGNORECASE,
)

_FAIL_RE = re.compile(
    r"#(\d+)\s+"
    r"(BUILD FAILED|TEST FAILED|TIMEOUT|NO_CODE|BENCH FAILED|"
    r"EXTRACTION_FAILED|LLM_ERROR|NO CODE)\s+"
    r"N/?A",
    re.IGNORECASE,
)


def parse_log(path: str) -> list[Iteration]:
    iterations: list[Iteration] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            m = _LINE_RE.search(line)
            if m:
                iterations.append(Iteration(
                    eval_num=int(m.group(1)),
                    status=m.group(2).lower().replace(" ", "_"),
                    score=float(m.group(3)),
                    best_so_far=float(m.group(4)),
                ))
                continue
            m = _FAIL_RE.search(line)
            if m:
                iterations.append(Iteration(
                    eval_num=int(m.group(1)),
                    status=m.group(2).lower().replace(" ", "_"),
                    score=None,
                    best_so_far=None,
                ))
    return iterations


# ── Rolling statistics ───────────────────────────────────────────────

def rolling_stats(scores: list[float], window: int = 8):
    """Compute rolling median, Q1, Q3, and coefficient of variation."""
    n = len(scores)
    medians = []
    q1s = []
    q3s = []
    covs = []  # coefficient of variation (%)
    for i in range(n):
        start = max(0, i - window + 1)
        w = scores[start:i + 1]
        medians.append(float(np.median(w)))
        q1s.append(float(np.percentile(w, 25)))
        q3s.append(float(np.percentile(w, 75)))
        mean = float(np.mean(w))
        std = float(np.std(w, ddof=0))
        covs.append(100.0 * std / mean if mean > 0 else 0.0)
    return medians, q1s, q3s, covs


# ── Phase detection (heuristic) ──────────────────────────────────────

def detect_phases(scores: list[float], eval_nums: list[int],
                  baseline: float) -> list[tuple[int, int, str]]:
    """
    Simple heuristic phase detection:
    - Exploration: first portion with high variance
    - Discovery: where we see a large jump (best-so-far drops significantly)
    - Refinement: tail where scores are stable around the best
    """
    n = len(scores)
    if n < 6:
        return [(eval_nums[0], eval_nums[-1], "Exploration")]

    # Find the biggest single-step improvement
    best = baseline
    best_improvements = []
    for i, s in enumerate(scores):
        if s < best:
            improvement = best - s
            best_improvements.append((i, improvement))
            best = s

    if not best_improvements:
        return [(eval_nums[0], eval_nums[-1], "Exploration")]

    # Discovery starts at the biggest improvement
    biggest_idx = max(best_improvements, key=lambda x: x[1])[0]

    # Exploration: start to just before discovery
    # Discovery: around the biggest jump (a few evals on each side)
    # Refinement: after discovery settles

    disc_start = max(0, biggest_idx - 2)
    disc_end = min(n - 1, biggest_idx + int(n * 0.15))

    # Refinement: last 40% of evaluations
    refine_start = max(disc_end + 1, int(n * 0.55))

    phases = []
    if disc_start > 0:
        phases.append((eval_nums[0], eval_nums[disc_start - 1], "Exploration"))
    phases.append((eval_nums[disc_start], eval_nums[min(disc_end, n - 1)], "Discovery"))
    if refine_start < n:
        phases.append((eval_nums[refine_start], eval_nums[-1], "Refinement"))

    return phases


# ── Main plotting ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Plot AutoPerf exploration dynamics (2-panel).")
    ap.add_argument("--log", required=True, help="Log file for a kernel")
    ap.add_argument("--baseline", type=float, required=True,
                    help="Baseline score (ns/op)")
    ap.add_argument("--kernel-name", default="matmul",
                    help="Kernel name for titles")
    ap.add_argument("--window", type=int, default=8,
                    help="Rolling window size")
    ap.add_argument("--output", "-o", default="plots/exploration.png",
                    help="Output image path")
    args = ap.parse_args()

    iters = parse_log(args.log)
    scored = [it for it in iters if it.score is not None]
    if not scored:
        print("ERROR: no scored iterations found in log.", file=sys.stderr)
        sys.exit(1)

    eval_nums = [it.eval_num for it in scored]
    scores = [it.score for it in scored]

    medians, q1s, q3s, covs = rolling_stats(scores, args.window)

    # Best-so-far trace
    best_trace = []
    best = args.baseline
    for s in scores:
        if s < best:
            best = s
        best_trace.append(best)

    # Detect phases
    phases = detect_phases(scores, eval_nums, args.baseline)

    # ── Figure ───────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5),
                                    height_ratios=[1.3, 1],
                                    sharex=True)
    fig.suptitle("AutoPerf: Exploration vs Exploitation Dynamics",
                 fontsize=13, fontweight="bold", y=0.98)

    # ── Panel 1: Search phases ───────────────────────────────────────
    ax1.set_facecolor("white")

    # Phase background bands (labels drawn after axis limits settle)
    for (x0, x1, phase_name) in phases:
        ax1.axvspan(x0 - 0.5, x1 + 0.5,
                    color=PHASE_COLORS.get(phase_name, "#f3f4f6"),
                    alpha=0.5, zorder=0)

    # Score spread band (IQR -> "Score spread")
    ax1.fill_between(eval_nums, q1s, q3s, alpha=0.25, color=CORAL,
                     label="Score spread", zorder=1)

    # Rolling median
    ax1.plot(eval_nums, medians, color=CORAL, linewidth=1.5,
             label=f"Rolling median", zorder=2)

    # Individual scores as dots
    ax1.scatter(eval_nums, scores, c=PURPLE, s=18, alpha=0.5, zorder=3,
                edgecolors="none")

    # Best-so-far step line
    ax1.step(eval_nums, best_trace, where="post", color=PURPLE,
             linewidth=2, label="Best so far", zorder=4)

    ax1.set_title(f"{args.kernel_name}: Search Phases", fontsize=10,
                  fontweight="bold", pad=8)
    ax1.set_ylabel("Latency (ns/op)", fontsize=9)
    ax1.tick_params(labelsize=8)

    # Format large numbers
    if args.baseline > 1000:
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    ax1.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

    # Re-draw phase labels at proper y position after axis limits settle
    ax1.set_ylim(bottom=0)
    y_top = ax1.get_ylim()[1]
    for (x0, x1, phase_name) in phases:
        mid = (x0 + x1) / 2
        ax1.text(mid, y_top * 0.95, phase_name, ha="center", va="top",
                 fontsize=8, fontstyle="italic",
                 color=PHASE_LABEL_COLORS.get(phase_name, GRAY))

    # ── Panel 2: Volatility ─────────────────────────────────────────
    ax2.set_facecolor("white")

    # Phase background bands (same as panel 1)
    for (x0, x1, phase_name) in phases:
        ax2.axvspan(x0 - 0.5, x1 + 0.5,
                    color=PHASE_COLORS.get(phase_name, "#f3f4f6"),
                    alpha=0.3, zorder=0)

    ax2.fill_between(eval_nums, 0, covs, alpha=0.3, color=CORAL, zorder=1)
    ax2.plot(eval_nums, covs, color=CORAL, linewidth=1.5, zorder=2)

    # Annotate low-variance tail
    if len(covs) > 5 and covs[-1] < 30:
        ax2.annotate("Low variance", xy=(eval_nums[-1], covs[-1]),
                     xytext=(-40, 8), textcoords="offset points",
                     fontsize=7, fontstyle="italic", color=CORAL)

    ax2.set_title("Search Volatility Over Time", fontsize=10,
                  fontweight="bold", pad=8)
    ax2.set_xlabel("Evaluation #", fontsize=9)
    ax2.set_ylabel("Coefficient of Variation (%)", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.set_ylim(bottom=0)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
