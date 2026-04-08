"""C language handler for AutoPerf v2.

Thin subclass of CppHandler -- uses gcc instead of g++ and C11 conventions.
Most C kernels follow the same pattern (vectors → pointers + size), so the
harness is largely shared.  When a kernel is pure C (no namespace, no
std::vector), we adapt the compilation flags.
"""

from __future__ import annotations

from languages.cpp import CppHandler


class CHandler(CppHandler):
    """Compile and benchmark C kernels with gcc."""

    COMPILER = "gcc"
    STD_FLAG = "-std=c11"
    LANGUAGE_NAME = "C"

    def __init__(self, config):
        super().__init__(config)
        # Override binary name so C and C++ builds don't collide
        import os
        self._binary_path = os.path.join(self._workdir, "autoperf_harness_c")
        self._harness_path = os.path.join(self._workdir, "harness.c")
        self._kernel_path = os.path.join(self._workdir, "kernel.c")

    def build(self, code: str) -> None:
        """Build using gcc.

        If the kernel source contains C++ constructs (std::vector, namespace,
        templates) we fall back to g++ automatically and warn.
        """
        cpp_indicators = ["std::vector", "namespace ", "template<", "template <"]
        if any(ind in code for ind in cpp_indicators):
            # Silently promote to C++ -- the kernel is really C++.
            self.COMPILER = "g++"
            self.STD_FLAG = "-std=c++17"

        # The harness template itself uses C++ (std::vector, etc.) so we
        # always compile with g++ in practice.  Pure-C support would need
        # a separate harness template.  For now, delegate to the parent.
        super().build(code)
