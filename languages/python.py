"""Python language handler for AutoPerf v2.

Build is a no-op (interpreted).  Testing runs the solution with assertions.
Benchmarking uses pyperf when available, falling back to timeit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

from languages.base import LanguageHandler


class PythonHandler(LanguageHandler):
    """Run and benchmark Python kernels."""

    def __init__(self, config):
        super().__init__(config)
        self._solution_path = os.path.join(self._workdir, "solution.py")
        self._test_path = os.path.join(self._workdir, "test_solution.py")
        self._bench_path = os.path.join(self._workdir, "bench_solution.py")
        self._python = sys.executable  # use the same interpreter

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_sandboxed(self, args: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a command under sandbox constraints (timeout only for Python)."""
        cmd = self._sandbox_prefix() + args
        return subprocess.run(cmd, capture_output=True, text=True, **kwargs)

    @staticmethod
    def _has_pyperf() -> bool:
        """Check whether pyperf is installed."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pyperf", "--version"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except (FileNotFoundError, OSError):
            return False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(self, code: str) -> None:
        """Write the solution to disk.  No compilation needed.

        We do a quick syntax check (``py_compile``) so errors surface early.
        """
        with open(self._solution_path, "w") as f:
            f.write(code)

        # Syntax check
        result = subprocess.run(
            [self._python, "-m", "py_compile", self._solution_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Python syntax check failed:\n{result.stderr}\n{result.stdout}"
            )

    def test(self) -> bool:
        """Import the solution and run basic smoke tests.

        The solution module must expose a callable named ``kernel``.
        We call it with a small list and check that it does not crash.
        If the solution defines a ``self_test()`` function we call that
        instead.
        """
        test_code = textwrap.dedent(f"""\
            import sys
            sys.path.insert(0, {self._workdir!r})
            import solution

            # If the solution has its own self_test, use it.
            if hasattr(solution, 'self_test'):
                solution.self_test()
                sys.exit(0)

            # Otherwise, smoke-test: just call kernel() if it exists.
            if hasattr(solution, 'kernel'):
                import array
                data = list(range(1024))
                try:
                    result = solution.kernel(data)
                except Exception as exc:
                    print(f"kernel() raised: {{exc}}", file=sys.stderr)
                    sys.exit(1)
                sys.exit(0)

            print("No kernel() or self_test() found in solution.py", file=sys.stderr)
            sys.exit(1)
        """)
        with open(self._test_path, "w") as f:
            f.write(test_code)

        result = self._run_sandboxed([self._python, self._test_path])
        if result.returncode != 0:
            print(f"[AutoPerf] Python test FAILED (exit {result.returncode})")
            if result.stderr:
                print(result.stderr)
            return False
        return True

    def benchmark(self) -> float:
        """Benchmark the solution and return ns/op.

        Strategy:
          1. Try pyperf (precise, handles JIT warm-up, outputs JSON).
          2. Fall back to timeit (always available).
        """
        if not os.path.isfile(self._solution_path):
            raise RuntimeError("Solution not found. Call build() first.")

        if self._has_pyperf():
            return self._bench_pyperf()
        return self._bench_timeit()

    # ------------------------------------------------------------------
    # Benchmark backends
    # ------------------------------------------------------------------

    def _bench_pyperf(self) -> float:
        """Use pyperf for high-quality benchmarking."""
        result_json = os.path.join(self._workdir, "pyperf_result.json")
        setup = (
            f"import sys; sys.path.insert(0, {self._workdir!r}); "
            f"from solution import kernel; "
            f"data = list(range(4096))"
        )
        stmt = "kernel(data)"

        result = self._run_sandboxed([
            self._python, "-m", "pyperf", "timeit",
            "-s", setup,
            stmt,
            "--json-output", result_json,
        ])
        if result.returncode != 0:
            raise RuntimeError(
                f"pyperf failed (exit {result.returncode}):\n"
                f"stderr: {result.stderr}\nstdout: {result.stdout}"
            )

        with open(result_json) as f:
            data = json.load(f)

        # pyperf JSON: benchmarks[0].runs[*].values -- mean of all values
        values = []
        for run in data.get("benchmarks", [{}])[0].get("runs", []):
            values.extend(run.get("values", []))
        if not values:
            raise RuntimeError("pyperf produced no timing values")

        mean_sec = sum(values) / len(values)
        return mean_sec * 1e9  # convert to ns/op

    def _bench_timeit(self) -> float:
        """Fall back to the stdlib timeit module."""
        bench_code = textwrap.dedent(f"""\
            import sys, timeit, json
            sys.path.insert(0, {self._workdir!r})
            from solution import kernel

            data = list(range(4096))

            # Auto-range: find an iteration count that takes >= 0.5s
            timer = timeit.Timer("kernel(data)", globals={{"kernel": kernel, "data": data}})
            number, total = timer.autorange()
            # Run 5 repetitions
            times = timer.repeat(repeat=5, number=number)
            best = min(times) / number  # seconds per call
            ns_per_op = best * 1e9
            print(json.dumps({{"ns_per_op": ns_per_op, "number": number, "repeats": 5}}))
        """)
        with open(self._bench_path, "w") as f:
            f.write(bench_code)

        result = self._run_sandboxed([self._python, self._bench_path])
        if result.returncode != 0:
            raise RuntimeError(
                f"timeit benchmark failed (exit {result.returncode}):\n"
                f"stderr: {result.stderr}\nstdout: {result.stdout}"
            )

        stdout = result.stdout.strip()
        try:
            data = json.loads(stdout)
            return float(data["ns_per_op"])
        except (json.JSONDecodeError, KeyError) as exc:
            raise RuntimeError(
                f"Could not parse timeit output: {exc}\n"
                f"Raw stdout: {stdout!r}"
            ) from exc
