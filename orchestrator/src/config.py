import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CPP_ROOT = os.path.join(REPO_ROOT, "cpp")
BUILD_DIR = os.path.join(REPO_ROOT, "build")
RUNS_DIR = os.path.join(REPO_ROOT, "orchestrator", "runs")
BEST_DIR = os.path.join(REPO_ROOT, "orchestrator", "best")
PROMPTS_DIR = os.path.join(REPO_ROOT, "orchestrator", "prompts")
ENV_PATH = os.path.join(REPO_ROOT, "orchestrator", ".env")
ORCH_SRC_DIR = os.path.join(REPO_ROOT, "orchestrator", "src")

# Central kernel catalog
KERNEL_TYPES = ["matvec", "matmul", "reduce", "search", "custom", "axpy"]
KERNEL_BASENAMES = {
    "matvec": "kernel_matvec.cpp",
    "matmul": "kernel_matmul.cpp",
    "reduce": "kernel_reduce.cpp",
    "search": "kernel_search.cpp",
    "custom": "kernel_custom.cpp",
    "axpy": "kernel_axpy.cpp",
}

def kernel_file_path(kernel_type: str) -> str:
    basename = KERNEL_BASENAMES.get(kernel_type)
    if not basename:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    return os.path.join(CPP_ROOT, "src", basename)

# Timeouts (seconds)
CONFIGURE_TIMEOUT_S = 300
BUILD_TIMEOUT_S = 300
TEST_TIMEOUT_S = 60
BENCH_TIMEOUT_S = 60
