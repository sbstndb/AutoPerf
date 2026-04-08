#!/usr/bin/env python3
"""
Plot AutoPerf Score Distributions: violin + strip plots.

Parses AutoPerf stdout log format:
    [d1/b0] #1  IMPROVED  3.8 ns/op  (best: 5.3)
    [d1/b1] #2  REGRESSION  4.5 ns/op  (best: 3.8)
    [d1/b2] #3  BUILD FAILED  N/A

Usage:
    python scripts/plot_score_distributions.py \
        --log-pow pow_log.txt --log-matmul matmul_log.txt \
        --baseline-pow 5.0 --baseline-matmul 126460 \
        --human-baseline 2.75 \
        --output plots/distributions.png
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

# ── Colours ──────────────────────────────────────────────────────────
GREEN = "#22c55e"
RED = "#ef4444"
PURPLE = "#6d28d9"
CORAL = "#D97757"
GRAY = "#9ca3af"
LIGHT_PURPLE = "#c4b5fd"

# ── Data structures ──────────────────────────────────────────────────

class Iteration(NamedTuple):
    eval_num: int
    status: str
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


# ── Single panel plotter ─────────────────────────────────────────────

def _plot_panel(ax, iters: list[Iteration], baseline: float,
                title: str, human_baseline: float | None = None):
    """Draw one violin + strip panel."""
    scored = [it for it in iters if it.score is not None]
    if not scored:
        ax.text(0.5, 0.5, "No scored iterations", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color=GRAY)
        return

    scores = [it.score for it in scored]
    speedups = [baseline / s for s in scores]

    # Violin
    parts = ax.violinplot(speedups, positions=[1], showmeans=False,
                          showmedians=False, showextrema=False,
                          widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor(LIGHT_PURPLE)
        pc.set_edgecolor(PURPLE)
        pc.set_alpha(0.4)
        pc.set_linewidth(0.8)

    # Strip (jittered dots)
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(speedups))
    ax.scatter(1 + jitter, speedups, c=CORAL, s=22, alpha=0.6,
               edgecolors="white", linewidths=0.3, zorder=3)

    # Baseline line at 1x
    ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=1, zorder=1)
    ax.annotate("Baseline (1x)", xy=(1.35, 1.0), fontsize=7,
                color=GRAY, fontstyle="italic", va="center")

    # Human baseline (pow_int only)
    if human_baseline is not None:
        human_speedup = baseline / human_baseline
        ax.axhline(human_speedup, color=CORAL, linestyle="--",
                   linewidth=1.2, zorder=2)
        ax.annotate(f"Human ({human_speedup:.2f}x)",
                    xy=(1.35, human_speedup), fontsize=7,
                    color=CORAL, fontstyle="italic", va="center")

    # Median annotation
    med = float(np.median(speedups))
    ax.annotate(f"Median: {med:.2f}x",
                xy=(1, med), xytext=(25, -10),
                textcoords="offset points", fontsize=8,
                color=PURPLE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.5))

    # Best annotation
    best_speedup = max(speedups)
    ax.annotate(f"Best: {best_speedup:.1f}x",
                xy=(1, best_speedup), xytext=(25, 5),
                textcoords="offset points", fontsize=8,
                color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.5))

    # Stats in top-left
    n_beat = sum(1 for s in speedups if s > 1.0)
    pct_beat = 100.0 * n_beat / len(speedups)
    stats_text = f"{pct_beat:.0f}% beat baseline"
    if human_baseline is not None:
        n_beat_human = sum(1 for s in speedups if s > baseline / human_baseline)
        pct_human = 100.0 * n_beat_human / len(speedups)
        stats_text += f"\n{pct_human:.0f}% beat human"
    if best_speedup > 1:
        stats_text += f"\nBest: {best_speedup:.1f}x faster"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=7.5, color="#374151")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("Speedup vs Baseline", fontsize=9)
    ax.set_xticks([1])
    ax.set_xticklabels(["AutoPerf\nattempts"], fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0.3, 1.7)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Plot AutoPerf score distributions (violin + strip).")
    ap.add_argument("--log-pow", help="Log file for pow_int kernel")
    ap.add_argument("--log-matmul", help="Log file for matmul kernel")
    ap.add_argument("--baseline-pow", type=float, default=5.0,
                    help="Baseline score for pow_int (ns/op)")
    ap.add_argument("--baseline-matmul", type=float, default=126460,
                    help="Baseline score for matmul (ns/op)")
    ap.add_argument("--human-baseline", type=float, default=None,
                    help="Human-optimized baseline for pow_int (ns/op)")
    ap.add_argument("--output", "-o", default="plots/distributions.png",
                    help="Output image path")
    args = ap.parse_args()

    if not args.log_pow and not args.log_matmul:
        ap.error("Provide at least one of --log-pow or --log-matmul")

    n_panels = sum(1 for x in [args.log_pow, args.log_matmul] if x)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
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
                    "matmul")

    fig.suptitle("AutoPerf: Distribution of LLM-Generated Optimizations",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
