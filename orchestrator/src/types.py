class OptResult:
    """Result of a single optimization attempt."""
    def __init__(self, id, parent_id=None, success=False, compile_ok=False, tests_ok=False,
                 benchmark_ns_per_item=None, kernel_code="", log="",
                 failure_stage=None, model_used=None, prompt_used=None):
        self.id = id
        self.parent_id = parent_id
        self.success = success
        self.compile_ok = compile_ok
        self.tests_ok = tests_ok
        self.benchmark_ns_per_item = benchmark_ns_per_item
        self.kernel_code = kernel_code
        self.log = log
        self.failure_stage = failure_stage
        self.model_used = model_used
        self.prompt_used = prompt_used

    @property
    def score(self):
        return self.benchmark_ns_per_item or float("inf")


class Baseline:
    """Details of the baseline performance."""
    def __init__(self, success=False, score=0.0, log=""):
        self.success = success
        self.score = score
        self.log = log
