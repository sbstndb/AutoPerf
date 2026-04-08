from dataclasses import dataclass


@dataclass
class AutoPerfConfig:
    kernel_path: str                    # path to source file to optimize
    language: str = "cpp"               # cpp, c, python
    model: str = "openrouter/anthropic/claude-sonnet-4"
    api_key: str | None = None          # from env if not set
    temperature: float = 0.3
    beam_width: int = 3                 # K in beam search
    branching_factor: int = 3           # B per candidate
    max_depth: int = 3                  # search depth
    compiler_flags: str = "-O3 -march=native"
    pin_core: int | None = 0
    timeout_sec: int = 30
    max_memory_mb: int = 1024
