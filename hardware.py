"""Detect hardware characteristics for inclusion in LLM optimisation prompts."""

from __future__ import annotations

import platform
import subprocess


def get_hardware_info() -> str:
    """Return a multi-line summary of CPU model, caches, and SIMD support.

    Best-effort: silently degrades on non-Linux or when /proc is unavailable.
    """
    info: list[str] = [f"Architecture: {platform.machine()}"]

    # ---- CPU model from /proc/cpuinfo ----
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info.append(f"CPU: {line.split(':', 1)[1].strip()}")
                    break
    except (OSError, IndexError):
        pass

    # ---- Core count ----
    try:
        import os

        logical = os.cpu_count()
        if logical:
            info.append(f"Logical cores: {logical}")
    except Exception:
        pass

    # ---- Cache sizes from lscpu ----
    try:
        result = subprocess.run(
            ["lscpu"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "cache" in line.lower():
                    info.append(line.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # ---- SIMD capabilities ----
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read().lower()
            simd_features: list[str] = []
            # Order from newest / widest to oldest / narrowest.
            for feat in ("avx512", "avx2", "avx", "sse4_2", "sse4_1", "ssse3", "neon"):
                if feat in content:
                    simd_features.append(feat.upper())
            if simd_features:
                info.append(f"SIMD: {', '.join(simd_features)}")
    except OSError:
        pass

    # ---- Memory ----
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    gb = kb / (1024 * 1024)
                    info.append(f"Memory: {gb:.1f} GiB")
                    break
    except (OSError, ValueError, IndexError):
        pass

    return "\n".join(info)
