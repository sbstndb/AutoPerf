"""Sandboxed command execution with timeout, memory limits, and CPU pinning."""

import subprocess
import os
import platform
import shlex


class SandboxError(Exception):
    """Raised when a sandboxed command fails due to infrastructure issues."""

    def __init__(self, message: str, returncode: int = -1, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_sandboxed(
    command: list[str],
    timeout_sec: int = 30,
    max_memory_mb: int = 1024,
    pin_core: int | None = 0,
    cwd: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a command with timeout, memory limits, and optional CPU pinning.

    Args:
        command: Command and arguments as a list of strings.
        timeout_sec: Maximum wall-clock seconds before the process is killed.
        max_memory_mb: Virtual memory cap in MiB (Linux only, 0 to disable).
        pin_core: CPU core to pin to via taskset (Linux only, None to disable).
        cwd: Working directory for the subprocess.

    Returns:
        subprocess.CompletedProcess with stdout/stderr captured as text.

    Raises:
        SandboxError: On timeout, OS-level failure, or missing executable.
    """
    if not command:
        raise SandboxError("Empty command list")

    is_linux = platform.system() == "Linux"

    try:
        if is_linux:
            # Build the shell command with optional CPU pinning and memory limit.
            parts: list[str] = []

            if max_memory_mb:
                # ulimit -v is in KiB
                parts.append(f"ulimit -v {max_memory_mb * 1024}")

            # Quote each token safely
            cmd_str = " ".join(shlex.quote(c) for c in command)

            if pin_core is not None:
                cmd_str = f"taskset -c {int(pin_core)} {cmd_str}"

            parts.append(cmd_str)
            shell_cmd = "; ".join(parts)

            result = subprocess.run(
                ["bash", "-c", shell_cmd],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=cwd,
            )
        else:
            # Non-Linux: no ulimit / taskset support; run directly.
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=cwd,
            )

        return result

    except subprocess.TimeoutExpired as exc:
        raise SandboxError(
            f"Command timed out after {timeout_sec}s: {' '.join(command)}",
            returncode=-1,
            stdout=exc.stdout or "" if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace"),
            stderr=exc.stderr or "" if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace"),
        ) from exc

    except FileNotFoundError as exc:
        raise SandboxError(
            f"Executable not found: {command[0]}",
            returncode=-1,
        ) from exc

    except PermissionError as exc:
        raise SandboxError(
            f"Permission denied running: {command[0]}",
            returncode=-1,
        ) from exc

    except OSError as exc:
        raise SandboxError(
            f"OS error running command: {exc}",
            returncode=-1,
        ) from exc
