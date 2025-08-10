import random
from orchestrator.src.types import OptResult

class ParentSelection:
    """Determines which successful result to use as the basis for the next iteration.

    Note: hierarchical mode does not use this, but we keep minimal support
    to avoid breaking future extensions.
    """
    def __init__(self, mode: str):
        self.mode = mode

    def choose(self, results):
        successes = [r for r in results if r.success]
        if not successes:
            return None
        if self.mode == "best":
            return min(successes, key=lambda r: r.score)
        if self.mode == "random_success":
            return random.choice(successes)
        return successes[0]


class Optimizer:
    """
    Manages the optimization process by tracking results and selecting parent kernels.
    """
    def __init__(self, parent_selection_mode: str, baseline_kernel: str, baseline_score: float):
        self._parent_strategy = ParentSelection(mode=parent_selection_mode)
        self.history: list[OptResult] = []
        self.best_result: OptResult | None = None
        self._baseline_kernel = baseline_kernel
        self._baseline_score = baseline_score

    def add_result(self, result):
        """Adds a new optimization result to the history."""
        self.history.append(result)
        if result.success:
            if self.best_result is None or result.score < self.best_result.score:
                self.best_result = result

    # get_next_parent not used in hierarchical mode; keep for potential future use

    def get_result_by_id(self, result_id):
        """Finds a result in the history by its ID."""
        for result in self.history:
            if result.id == result_id:
                return result
        return None
