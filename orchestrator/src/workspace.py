import json
import os
from datetime import datetime

from orchestrator.src.config import RUNS_DIR, BEST_DIR, KERNEL_BASENAMES


class Workspace:
    """Manages the run directory and artifacts."""

    def __init__(self, name = None, kernel_type: str = "vector_add"):
        self._kernel_type = kernel_type
        prefix = name if name else f"run_{kernel_type}"
        when = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = os.path.join(RUNS_DIR, f"{when}_{prefix}")
        os.makedirs(self._run_dir, exist_ok=True)
        self._log_dir = os.path.join(self._run_dir, "logs")
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(BEST_DIR, exist_ok=True)

        self._symlink_current()

    @property
    def path(self):
        return self._run_dir

    def _symlink_current(self):
        current_link = os.path.join(RUNS_DIR, "CURRENT")
        if os.path.exists(current_link) or os.path.islink(current_link):
            os.unlink(current_link)
        os.symlink(self._run_dir, current_link)

    def save_baseline_log(self, log):
        with open(os.path.join(self._log_dir, "baseline.log"), 'w', encoding='utf-8') as f:
            f.write(log)

    def save_best_kernel(self, kernel_code):
        file_name = KERNEL_BASENAMES.get(self._kernel_type, "kernel.cpp")
        with open(os.path.join(BEST_DIR, file_name), 'w', encoding='utf-8') as f:
            f.write(kernel_code)

    def save_iteration(self, iteration, result):
        iter_dir = os.path.join(self._run_dir, f"iter_{iteration:03d}")
        os.makedirs(iter_dir, exist_ok=True)
        
        with open(os.path.join(iter_dir, "kernel.cpp"), 'w', encoding='utf-8') as f:
            f.write(result.kernel_code)
        with open(os.path.join(iter_dir, "log.txt"), 'w', encoding='utf-8') as f:
            f.write(result.log)
        
        meta = {
            "success": result.success,
            "score": result.score,
            "failure_stage": result.failure_stage,
            "model_used": result.model_used,
            "prompt_used": result.prompt_used,
        }
        with open(os.path.join(iter_dir, "meta.json"), 'w', encoding='utf-8') as f:
            f.write(json.dumps(meta, indent=2))

    def get_cpp_project_path(self, iteration):
        iter_dir = os.path.join(self._run_dir, f"iter_{iteration:03d}")
        cpp_project_path = os.path.join(iter_dir, "cpp_project")
        os.makedirs(cpp_project_path, exist_ok=True)
        return cpp_project_path
