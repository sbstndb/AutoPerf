import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime

from llm import call_llm, extract_code
from hardware import get_hardware_info
from history import OptimizationHistory, Attempt
from ui import UI


@dataclass
class Candidate:
    code: str
    score: float  # lower is better (ns/op)
    strategy: str  # LLM's explanation
    depth: int
    parent_id: int | None = None
    id: int = 0


def _create_run_dir() -> str:
    """Create a timestamped directory under runs/ for this optimization run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _save_iteration(run_dir: str, iteration: int, code: str, metadata: dict, language: str = "cpp") -> None:
    """Save generated code and metadata for one iteration."""
    ext = {"cpp": "cpp", "c": "c", "python": "py"}.get(language, "txt")
    code_path = os.path.join(run_dir, f"iter_{iteration}.{ext}")
    meta_path = os.path.join(run_dir, f"iter_{iteration}.json")

    with open(code_path, "w") as f:
        f.write(code)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _save_summary(run_dir: str, best: Candidate, baseline_score: float, elapsed: float) -> None:
    """Save final summary of the optimization run."""
    summary = {
        "best_score": best.score,
        "baseline_score": baseline_score,
        "speedup": baseline_score / best.score if best.score > 0 else 0,
        "best_id": best.id,
        "best_depth": best.depth,
        "strategy": best.strategy[:500],
        "elapsed_seconds": round(elapsed, 1),
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def beam_search(config, language_handler, system_prompt, user_prompt_template, baseline_code, baseline_score):
    """
    Beam search over LLM-generated optimizations.

    At each depth level:
    1. For each candidate in the beam, generate B variants via LLM
    2. Build, test, benchmark each variant
    3. Keep the top-K candidates (by benchmark score)
    4. Repeat for max_depth levels

    Returns the best candidate found.
    """
    ui = UI()
    history = OptimizationHistory()
    history.baseline_score = baseline_score
    hw_info = get_hardware_info()

    candidates = [Candidate(code=baseline_code, score=baseline_score, strategy="baseline", depth=0)]
    best = candidates[0]
    next_id = 1

    compiler = "g++" if config.language == "cpp" else ("gcc" if config.language == "c" else "python")

    # Extract dataset info from harness if --show-dataset
    dataset_info = ""
    if config.show_dataset:
        harness_path = config.kernel_path.replace(".cpp", "_harness.cpp")
        if os.path.isfile(harness_path):
            with open(harness_path) as f:
                harness_src = f.read()
            # Extract arrays from harness source
            import re
            arrays = re.findall(r'(?:const\s+\w+\s+\w+\[\]\s*=\s*\{[^}]+\})', harness_src)
            if arrays:
                dataset_info = "=== BENCHMARK DATASET ===\nThe benchmark measures average ns/op over these inputs:\n"
                for arr in arrays:
                    dataset_info += f"  {arr}\n"
                dataset_info += "Your optimization will be scored on the average across ALL these inputs.\n\n"
        if not dataset_info:
            ui.info("--show-dataset: could not extract dataset from harness")

    # Create run directory for saving per-iteration data
    run_dir = _create_run_dir()
    ui.info(f"Run data will be saved to: {run_dir}")

    start_time = time.time()

    for depth in range(1, config.max_depth + 1):
        next_candidates = []
        for parent in candidates:
            for branch in range(config.branching_factor):
                elapsed = time.time() - start_time

                # Get assembly for the parent if available
                asm = language_handler.get_assembly(parent.code) if hasattr(language_handler, 'get_assembly') else None
                if asm:
                    assembly_section = (
                        f"=== ASSEMBLY (hot function, Intel syntax) ===\n{asm}\n\n"
                    )
                else:
                    assembly_section = ""

                # Format prompt with all required fields
                user_prompt = user_prompt_template.format(
                    language=config.language,
                    hardware_info=hw_info,
                    compiler=compiler,
                    flags=config.compiler_flags,
                    score=f"{parent.score:.2f}",
                    history=history.format_for_llm(),
                    code=parent.code,
                    assembly_section=assembly_section,
                    dataset_section=dataset_info,
                )

                # Call LLM
                try:
                    response = call_llm(system_prompt, user_prompt, config.model, config.temperature)
                except Exception as e:
                    ui.error(f"LLM call failed: {e}")
                    history.add(Attempt(next_id, "LLM call failed", None, "llm_error", str(e)))
                    next_id += 1
                    continue

                new_code = extract_code(response, config.language)

                if new_code is None:
                    ui.iteration(depth, branch, next_id, "no_code",
                                 parent_id=parent.id, depth_level=depth, elapsed=elapsed)
                    history.add(Attempt(next_id, "No code extracted", None, "extraction_failed", None))
                    next_id += 1
                    continue

                # Extract strategy (text before code block)
                strategy = response.split("```")[0].strip()[-500:]

                # Build with retry on failure
                build_ok = False
                retry_fix = False
                try:
                    language_handler.build(new_code)
                    build_ok = True
                except Exception as e:
                    # Retry: send error back to LLM
                    for retry in range(2):
                        fix_prompt = (
                            f"The following code failed to compile:\n\n"
                            f"```cpp\n{new_code}\n```\n\n"
                            f"Compiler error:\n{str(e)[:500]}\n\n"
                            f"Fix this compilation error:\n"
                            f"Fix the error and return the corrected code."
                        )
                        try:
                            fix_response = call_llm(system_prompt, fix_prompt, config.model, config.temperature)
                        except Exception:
                            break
                        fixed_code = extract_code(fix_response, config.language)
                        if fixed_code is None:
                            break
                        new_code = fixed_code
                        try:
                            language_handler.build(new_code)
                            build_ok = True
                            retry_fix = True
                            ui.info(f"  Build fixed on retry {retry + 1}")
                            break  # fixed!
                        except Exception as e2:
                            e = e2  # update error for next retry

                if not build_ok:
                    ui.iteration(depth, branch, next_id, "build_failed",
                                 parent_id=parent.id, depth_level=depth, elapsed=elapsed)
                    history.add(Attempt(next_id, strategy[:100], None, "build_failed", str(e)[:200]))
                    _save_iteration(run_dir, next_id, new_code, {
                        "id": next_id, "score": None, "status": "build_failed",
                        "strategy": strategy[:200], "parent_id": parent.id, "depth": depth,
                        "retry_fix": retry_fix,
                    }, config.language)
                    next_id += 1
                    continue

                if not language_handler.test():
                    ui.iteration(depth, branch, next_id, "test_failed",
                                 parent_id=parent.id, depth_level=depth, elapsed=elapsed)
                    history.add(Attempt(next_id, strategy[:100], None, "test_failed", None))
                    _save_iteration(run_dir, next_id, new_code, {
                        "id": next_id, "score": None, "status": "test_failed",
                        "strategy": strategy[:200], "parent_id": parent.id, "depth": depth,
                        "retry_fix": retry_fix,
                    }, config.language)
                    next_id += 1
                    continue

                try:
                    score = language_handler.benchmark()
                except Exception as e:
                    ui.iteration(depth, branch, next_id, "bench_failed",
                                 parent_id=parent.id, depth_level=depth, elapsed=elapsed)
                    history.add(Attempt(next_id, strategy[:100], None, "bench_failed", str(e)[:200]))
                    _save_iteration(run_dir, next_id, new_code, {
                        "id": next_id, "score": None, "status": "bench_failed",
                        "strategy": strategy[:200], "parent_id": parent.id, "depth": depth,
                        "retry_fix": retry_fix,
                    }, config.language)
                    next_id += 1
                    continue

                candidate = Candidate(
                    code=new_code, score=score, strategy=strategy,
                    depth=depth, parent_id=parent.id, id=next_id,
                )
                next_id += 1

                status = "improved" if score < best.score else "regression"
                ui.iteration(depth, branch, candidate.id, status, score, best.score,
                             parent_id=parent.id, depth_level=depth,
                             retry_fix=retry_fix, elapsed=elapsed)
                history.add(Attempt(candidate.id, strategy[:100], score, status, None))

                # Save iteration data
                _save_iteration(run_dir, candidate.id, new_code, {
                    "id": candidate.id, "score": score, "status": status,
                    "strategy": strategy[:500], "parent_id": parent.id, "depth": depth,
                    "retry_fix": retry_fix,
                }, config.language)

                next_candidates.append(candidate)

                if score < best.score:
                    best = candidate

        if not next_candidates:
            ui.warn(f"Depth {depth}: all branches failed, stopping search.")
            break

        # Keep top-K
        next_candidates.sort(key=lambda c: c.score)
        candidates = next_candidates[:config.beam_width]

    # Save final summary
    total_elapsed = time.time() - start_time
    _save_summary(run_dir, best, baseline_score, total_elapsed)
    ui.info(f"Run summary saved to: {run_dir}/summary.json")

    return best
