from dataclasses import dataclass

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

    for depth in range(1, config.max_depth + 1):
        next_candidates = []
        for parent in candidates:
            for branch in range(config.branching_factor):
                # Format prompt with all required fields
                user_prompt = user_prompt_template.format(
                    language=config.language,
                    hardware_info=hw_info,
                    compiler=compiler,
                    flags=config.compiler_flags,
                    score=f"{parent.score:.2f}",
                    history=history.format_for_llm(),
                    code=parent.code,
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
                    ui.iteration(depth, branch, next_id, "no_code")
                    history.add(Attempt(next_id, "No code extracted", None, "extraction_failed", None))
                    next_id += 1
                    continue

                # Extract strategy (text before code block)
                strategy = response.split("```")[0].strip()[-500:]

                # Build, test, benchmark
                try:
                    language_handler.build(new_code)
                except Exception as e:
                    ui.iteration(depth, branch, next_id, "build_failed")
                    history.add(Attempt(next_id, strategy[:100], None, "build_failed", str(e)[:200]))
                    next_id += 1
                    continue

                if not language_handler.test():
                    ui.iteration(depth, branch, next_id, "test_failed")
                    history.add(Attempt(next_id, strategy[:100], None, "test_failed", None))
                    next_id += 1
                    continue

                try:
                    score = language_handler.benchmark()
                except Exception as e:
                    ui.iteration(depth, branch, next_id, "bench_failed")
                    history.add(Attempt(next_id, strategy[:100], None, "bench_failed", str(e)[:200]))
                    next_id += 1
                    continue

                candidate = Candidate(
                    code=new_code, score=score, strategy=strategy,
                    depth=depth, parent_id=parent.id, id=next_id,
                )
                next_id += 1

                status = "improved" if score < best.score else "regression"
                ui.iteration(depth, branch, candidate.id, status, score, best.score)
                history.add(Attempt(candidate.id, strategy[:100], score, status, None))

                next_candidates.append(candidate)

                if score < best.score:
                    best = candidate

        if not next_candidates:
            ui.warn(f"Depth {depth}: all branches failed, stopping search.")
            break

        # Keep top-K
        next_candidates.sort(key=lambda c: c.score)
        candidates = next_candidates[:config.beam_width]

    return best
