"""Abstract base class for all language handlers."""

from abc import ABC, abstractmethod
import os


class LanguageHandler(ABC):
    """Interface that every language handler must implement.

    A handler knows how to:
      1. build  - compile or prepare the code
      2. test   - verify correctness against known references
      3. benchmark - measure wall-clock performance (ns/op)
    """

    def __init__(self, config):
        self.config = config
        self._workdir = "/tmp/autoperf_work"
        os.makedirs(self._workdir, exist_ok=True)

    # -- helpers shared across handlers ----------------------------------------

    def _sandbox_prefix(self) -> list[str]:
        """Return a command prefix that enforces resource limits.

        Uses timeout(1), ulimit-via-bash, and optionally taskset for core
        pinning.  Callers prepend this to their subprocess argv.
        """
        parts: list[str] = []

        # Timeout
        timeout = getattr(self.config, "timeout_sec", 30)
        parts += ["timeout", str(timeout)]

        # Core pinning
        pin = getattr(self.config, "pin_core", None)
        if pin is not None:
            parts += ["taskset", "-c", str(pin)]

        return parts

    def _ulimit_preexec(self):
        """Return a preexec_fn that sets memory limits via setrlimit.

        Suitable for passing to subprocess.Popen / subprocess.run.
        """
        max_mem = getattr(self.config, "max_memory_mb", 1024)

        def _set_limits():
            import resource
            mem_bytes = max_mem * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

        return _set_limits

    # -- abstract interface ----------------------------------------------------

    @abstractmethod
    def build(self, code: str) -> None:
        """Compile / prepare *code*.  Raise on failure."""
        ...

    @abstractmethod
    def test(self) -> bool:
        """Run correctness tests.  Return True if all pass."""
        ...

    @abstractmethod
    def benchmark(self) -> float:
        """Run the benchmark.  Return ns/op (lower is better)."""
        ...
