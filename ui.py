"""Console output for AutoPerf -- Rich when available, plain-text fallback."""

from __future__ import annotations

import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class UI:
    """Thin presentation layer for AutoPerf CLI output."""

    def __init__(self) -> None:
        self.console = Console(stderr=True) if HAS_RICH else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _print(self, msg: str, style: str | None = None) -> None:
        if self.console and style:
            self.console.print(msg, style=style)
        elif self.console:
            self.console.print(msg)
        else:
            print(msg, file=sys.stderr)

    @staticmethod
    def _speedup_str(baseline: float, best: float) -> str:
        if baseline <= 0:
            return ""
        factor = baseline / best
        return f"{factor:.2f}x"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def banner(self) -> None:
        """Print startup banner."""
        text = (
            "AutoPerf v2  --  LLM-driven performance optimizer\n"
            "https://github.com/autoperf"
        )
        if self.console:
            self.console.print(
                Panel(text, title="[bold cyan]AutoPerf[/bold cyan]", border_style="cyan")
            )
        else:
            print("=" * 50, file=sys.stderr)
            print("  AutoPerf v2", file=sys.stderr)
            print("  LLM-driven performance optimizer", file=sys.stderr)
            print("=" * 50, file=sys.stderr)

    def baseline(self, score: float, kernel_path: str) -> None:
        """Print baseline measurement."""
        if self.console:
            self.console.print(
                f"[bold]Baseline:[/bold] [cyan]{score:.1f} ns/op[/cyan]  "
                f"[dim]({kernel_path})[/dim]"
            )
        else:
            print(f"Baseline: {score:.1f} ns/op  ({kernel_path})", file=sys.stderr)

    def iteration(
        self,
        depth: int,
        branch: int,
        candidate_id: int,
        status: str,
        score: float | None = None,
        best_score: float | None = None,
        parent_id: int | None = None,
        depth_level: int | None = None,
        retry_fix: bool = False,
        elapsed: float | None = None,
    ) -> None:
        """Print a single iteration result with colour coding."""
        label = f"[d{depth}/b{branch}] #{candidate_id}"
        score_str = f"{score:.1f} ns/op" if score is not None else "N/A"

        if status == "improved":
            style = "bold green"
            tag = "IMPROVED"
        elif status == "regression":
            style = "red"
            tag = "REGRESSION"
        elif status == "build_failed":
            style = "yellow"
            tag = "BUILD FAILED"
        elif status == "test_failed":
            style = "yellow"
            tag = "TEST FAILED"
        elif status == "timeout":
            style = "yellow"
            tag = "TIMEOUT"
        else:
            style = "dim"
            tag = status.upper()

        best_str = f"  (best: {best_score:.1f})" if best_score is not None else ""

        # Extra info: depth, parent, retry, elapsed
        extras = []
        if depth_level is not None:
            extras.append(f"depth={depth_level}")
        if parent_id is not None:
            extras.append(f"parent=#{parent_id}")
        if retry_fix:
            extras.append("retry-fixed")
        if elapsed is not None:
            mins, secs = divmod(int(elapsed), 60)
            extras.append(f"t={mins}m{secs:02d}s")
        extra_str = f"  ({', '.join(extras)})" if extras else ""

        if self.console:
            self.console.print(
                f"  {label}  [{style}]{tag}[/{style}]  {score_str}{best_str}{extra_str}"
            )
        else:
            print(f"  {label}  {tag}  {score_str}{best_str}{extra_str}", file=sys.stderr)

    def result(
        self,
        best_score: float,
        baseline_score: float,
        strategy: str,
        output_path: str,
    ) -> None:
        """Print final optimisation summary."""
        speedup = self._speedup_str(baseline_score, best_score)

        if self.console:
            table = Table(title="Optimization Result", border_style="green")
            table.add_column("Metric", style="bold")
            table.add_column("Value")
            table.add_row("Baseline", f"{baseline_score:.1f} ns/op")
            table.add_row("Best", f"[bold green]{best_score:.1f} ns/op[/bold green]")
            table.add_row("Speedup", f"[bold green]{speedup}[/bold green]")
            table.add_row("Strategy", strategy[:120])
            table.add_row("Output", output_path)
            self.console.print(table)
        else:
            print("", file=sys.stderr)
            print("=== Optimization Result ===", file=sys.stderr)
            print(f"  Baseline : {baseline_score:.1f} ns/op", file=sys.stderr)
            print(f"  Best     : {best_score:.1f} ns/op", file=sys.stderr)
            print(f"  Speedup  : {speedup}", file=sys.stderr)
            print(f"  Strategy : {strategy[:120]}", file=sys.stderr)
            print(f"  Output   : {output_path}", file=sys.stderr)
            print("===========================", file=sys.stderr)

    def error(self, msg: str) -> None:
        """Print an error message."""
        if self.console:
            self.console.print(f"[bold red]ERROR:[/bold red] {msg}")
        else:
            print(f"ERROR: {msg}", file=sys.stderr)

    def warn(self, msg: str) -> None:
        """Print a warning message."""
        if self.console:
            self.console.print(f"[bold yellow]WARN:[/bold yellow] {msg}")
        else:
            print(f"WARN: {msg}", file=sys.stderr)

    def info(self, msg: str) -> None:
        """Print an informational message."""
        if self.console:
            self.console.print(f"[dim]INFO:[/dim] {msg}")
        else:
            print(f"INFO: {msg}", file=sys.stderr)
