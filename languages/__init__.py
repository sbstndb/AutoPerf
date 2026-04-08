"""AutoPerf v2 language handlers.

Each handler knows how to build, test, and benchmark code in its language.
"""

from languages.base import LanguageHandler
from languages.cpp import CppHandler
from languages.c import CHandler
from languages.python import PythonHandler


def get_handler(config) -> LanguageHandler:
    """Factory: return the right handler for config.language."""
    handlers = {
        "cpp": CppHandler,
        "c++": CppHandler,
        "c": CHandler,
        "python": PythonHandler,
        "py": PythonHandler,
    }
    cls = handlers.get(config.language.lower())
    if cls is None:
        raise ValueError(
            f"Unsupported language: {config.language!r}. "
            f"Supported: {', '.join(sorted(handlers))}"
        )
    return cls(config)
