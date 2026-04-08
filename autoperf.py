#!/usr/bin/env python3
import argparse
import os
import sys

from config import AutoPerfConfig


def main():
    parser = argparse.ArgumentParser(description="AutoPerf v2 — LLM-driven code optimization")
    parser.add_argument("kernel", help="Path to the source file to optimize")
    parser.add_argument("--language", "-l", default="cpp", choices=["cpp", "c", "python"])
    parser.add_argument("--model", "-m", default=os.getenv("AUTOPERF_MODEL", "openrouter/anthropic/claude-sonnet-4"))
    parser.add_argument("--beam-width", "-K", type=int, default=3)
    parser.add_argument("--branching", "-B", type=int, default=3)
    parser.add_argument("--depth", "-D", type=int, default=3)
    parser.add_argument("--temperature", "-t", type=float, default=0.3)
    parser.add_argument("--pin-core", type=int, default=0)
    parser.add_argument("--flags", default="-O3 -march=native")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    config = AutoPerfConfig(
        kernel_path=args.kernel,
        language=args.language,
        model=args.model,
        beam_width=args.beam_width,
        branching_factor=args.branching,
        max_depth=args.depth,
        temperature=args.temperature,
        pin_core=args.pin_core,
        compiler_flags=args.flags,
        timeout_sec=args.timeout,
    )

    # Import language handler
    if config.language == "cpp":
        from languages.cpp import CppHandler
        handler = CppHandler(config)
    elif config.language == "c":
        from languages.c import CHandler
        handler = CHandler(config)
    elif config.language == "python":
        from languages.python import PythonHandler
        handler = PythonHandler(config)

    # Read kernel source
    with open(config.kernel_path) as f:
        baseline_code = f.read()

    # Load prompts
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    with open(os.path.join(prompts_dir, "system.txt")) as f:
        system_prompt = f.read()
    with open(os.path.join(prompts_dir, "optimize.txt")) as f:
        user_prompt_template = f.read()

    # Measure baseline
    print(f"[Baseline] Building and benchmarking {config.kernel_path}...")
    handler.build(baseline_code)
    if not handler.test():
        print("ERROR: Baseline fails correctness tests!")
        sys.exit(1)
    baseline_score = handler.benchmark()
    print(f"[Baseline] Score: {baseline_score:.2f} ns/op")

    # Run beam search
    from search import beam_search
    best = beam_search(config, handler, system_prompt, user_prompt_template, baseline_code, baseline_score)

    # Report
    speedup = baseline_score / best.score if best.score > 0 else 0
    print(f"\n{'='*60}")
    print(f"Best result: {best.score:.2f} ns/op ({speedup:.2f}x speedup)")
    print(f"Strategy: {best.strategy[:200]}")
    print(f"Depth: {best.depth}, ID: {best.id}")

    # Save optimized code
    out_path = config.kernel_path.replace(".", "_optimized.")
    with open(out_path, "w") as f:
        f.write(best.code)
    print(f"Optimized code saved to: {out_path}")


if __name__ == "__main__":
    main()
