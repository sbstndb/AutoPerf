#!/usr/bin/env python3
"""
Plot the AutoPerf Search Landscape: dual-panel scatter (pow_int + matmul).

Parses AutoPerf stdout log format:
    [d1/b0] #1  IMPROVED  3.8 ns/op  (best: 5.3)
    [d1/b1] #2  REGRESSION  4.5 ns/op  (best: 3.8)
    [d1/b2] #3  BUILD FAILED  N/A

Usage:
    python scripts/plot_search_landscape.py \
        --log-pow pow_log.txt --log-matmul matmul_log.txt \
        --baseline-pow 5.0 --baseline-matmul 126460 \
        --human-baseline 2.75 \
        --output plots/search_landscape.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Colours (site palette) ────────────────────────────────────────────
GREEN = "#22c55e"
RED = "#ef4444"
PURPLE = "#6d28d9"
CORAL = "#D97757"
GRAY = "#9ca3af"
LIGHT_GREEN_BG = "#f0fdf4"

# ── Data structures ──────────────────────────────────────────────────

class Iteration(NamedTuple):
    eval_num: int
    status: str        # improved | regression | build_failed | test_failed | ...
    score: float | None
    best_so_far: float | None


# ── Log parser ───────────────────────────────────────────────────────

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
    """Parse an AutoPerf log file and return per-iteration data."""
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


# ── Compute best-so-far trace ────────────────────────────────────────

def best_so_far_trace(iters: list[Iteration], baseline: float):
    """Return (eval_nums, best_scores) step trace including only scored iters."""
    xs, ys = [], []
    best = baseline
    for it in iters:
        if it.score is not None:
            if it.score < best:
                best = it.score
            xs.append(it.eval_num)
            ys.append(best)
    return xs, ys


# ── Single-panel plotter ─────────────────────────────────────────────

def _plot_panel(ax, iters: list[Iteration], baseline: float,
                title: str, human_baseline: float | None = None,
                large_scale: bool = False):
    """Draw one scatter panel on *ax*."""
    ax.set_facecolor("white")

    max_eval = max(it.eval_num for it in iters)

    # Scatter: green = improvement, red = regression
    for it in iters:
        if it.score is None:
            continue
        colour = GREEN if it.score < baseline else RED
        marker = "o"
        ax.scatter(it.eval_num, it.score, c=colour, s=36, marker=marker,
                   edgecolors="white", linewidths=0.4, zorder=3,
                   alpha=0.85)

    # Best-so-far step line
    bx, by = best_so_far_trace(iters, baseline)
    if bx:
        ax.step(bx, by, where="post", color=PURPLE, linewidth=2,
                label="Best so far", zorder=4)

    # Baseline annotation
    if large_scale:
        bl_label = f"Baseline {baseline:,.0f} ns"
    else:
        bl_label = f"Baseline {baseline:.2f} ns"
    ax.annotate(bl_label, xy=(1, baseline), xytext=(3, 4),
                textcoords="offset points", fontsize=7, color=GRAY,
                va="bottom")

    # Human-optimized baseline (pow_int only)
    if human_baseline is not None:
        ax.axhline(human_baseline, color=CORAL, linestyle="--", linewidth=1.2,
                   zorder=2)
        ax.annotate(f"Human opt. {human_baseline:.2f} ns",
                    xy=(max(it.eval_num for it in iters) * 0.55, human_baseline),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=7.5, color=CORAL, style="italic")

    # Best result annotation
    scored = [it for it in iters if it.score is not None]
    if scored:
        best_it = min(scored, key=lambda it: it.score)
        if large_scale:
            best_label = f"{best_it.score:,.0f} ns"
        else:
            best_label = f"{best_it.score:.1f} ns"
        ax.annotate(best_label,
                    xy=(best_it.eval_num, best_it.score),
                    xytext=(6, -6), textcoords="offset points",
                    fontsize=7.5, color=PURPLE, fontweight="bold")

        speedup = baseline / best_it.score
        ax.text(0.95, 0.05, f"{speedup:.1f}x speedup",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11, fontweight="bold", color=RED)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Evaluation #", fontsize=9)
    ax.set_ylabel("Latency (ns/op)", fontsize=9)
    ax.tick_params(labelsize=8)

    if large_scale:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    # Baseline line + green fill (drawn after data so axis limits are set)
    ax.axhline(baseline, color=GRAY, linestyle="--", linewidth=1, zorder=1)
    y_top = ax.get_ylim()[1]
    ax.fill_between(
        [0, max_eval + 1],
        baseline, y_top,
        color=LIGHT_GREEN_BG, alpha=0.4, zorder=0,
    )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Plot AutoPerf search landscape (dual-panel scatter).")
    ap.add_argument("--log-pow", help="Log file for pow_int kernel")
    ap.add_argument("--log-matmul", help="Log file for matmul kernel")
    ap.add_argument("--baseline-pow", type=float, default=5.0,
                    help="Baseline score for pow_int (ns/op)")
    ap.add_argument("--baseline-matmul", type=float, default=126460,
                    help="Baseline score for matmul (ns/op)")
    ap.add_argument("--human-baseline", type=float, default=None,
                    help="Human-optimized baseline for pow_int (ns/op)")
    ap.add_argument("--output", "-o", default="plots/search_landscape.png",
                    help="Output image path")
    args = ap.parse_args()

    if not args.log_pow and not args.log_matmul:
        ap.error("Provide at least one of --log-pow or --log-matmul")

    n_panels = sum(1 for x in [args.log_pow, args.log_matmul] if x)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    idx = 0
    if args.log_pow:
        iters_pow = parse_log(args.log_pow)
        _plot_panel(axes[idx], iters_pow, args.baseline_pow,
                    "pow_int", human_baseline=args.human_baseline)
        idx += 1
    if args.log_matmul:
        iters_mat = parse_log(args.log_matmul)
        _plot_panel(axes[idx], iters_mat, args.baseline_matmul,
                    "matmul", large_scale=True)

    # Legend (shared)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN,
               markersize=7, label="Improvement"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED,
               markersize=7, label="Regression"),
        Line2D([0], [0], color=PURPLE, linewidth=2, label="Best so far"),
        Line2D([0], [0], color=GRAY, linestyle="--", linewidth=1,
               label="Baseline"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, 0.98))

    fig.suptitle("AutoPerf Search Landscape", fontsize=13,
                 fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
