class OrchestratorError(Exception):
    """Base error for orchestrator operations."""


class OperationTimeout(OrchestratorError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] timeout: {message}")
        self.stage = stage


class ConfigureError(OrchestratorError):
    def __init__(self, message: str):
        super().__init__(message)
        self.stage = "configure"


class BuildError(OrchestratorError):
    def __init__(self, message: str):
        super().__init__(message)
        self.stage = "compile"


class TestError(OrchestratorError):
    def __init__(self, message: str):
        super().__init__(message)
        self.stage = "tests"


class BenchmarkError(OrchestratorError):
    def __init__(self, message: str):
        super().__init__(message)
        self.stage = "bench_fail"


class BenchmarkParseError(BenchmarkError):
    pass


class LLMError(OrchestratorError):
    def __init__(self, message: str):
        super().__init__(message)
        self.stage = "llm"

