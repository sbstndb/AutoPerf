from dataclasses import dataclass

from llm import call_llm, extract_code


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
    candidates = [Candidate(code=baseline_code, score=baseline_score, strategy="baseline", depth=0)]
    best = candidates[0]
    next_id = 1

    for depth in range(1, config.max_depth + 1):
        next_candidates = []
        for parent in candidates:
            for branch in range(config.branching_factor):
                # Format prompt with current code and history
                user_prompt = user_prompt_template.format(
                    code=parent.code,
                    score=parent.score,
                    depth=depth,
                    branch=branch,
                )

                # Call LLM
                response = call_llm(system_prompt, user_prompt, config.model, config.temperature)
                new_code = extract_code(response, config.language)

                if new_code is None:
                    continue  # LLM didn't return valid code

                # Extract strategy (text before code block)
                strategy = response.split("```")[0].strip()

                # Build, test, benchmark
                try:
                    language_handler.build(new_code)
                    if not language_handler.test():
                        continue  # correctness failed
                    score = language_handler.benchmark()
                except Exception:
                    continue  # build/test/bench failed

                candidate = Candidate(
                    code=new_code, score=score, strategy=strategy,
                    depth=depth, parent_id=parent.id, id=next_id,
                )
                next_id += 1
                next_candidates.append(candidate)

                if score < best.score:
                    best = candidate

        if not next_candidates:
            break  # all branches failed

        # Keep top-K
        next_candidates.sort(key=lambda c: c.score)
        candidates = next_candidates[:config.beam_width]

    return best
