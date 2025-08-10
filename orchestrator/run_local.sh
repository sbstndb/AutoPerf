#!/usr/bin/env bash
set -euo pipefail
set -x # Enable command tracing

echo "Starting AutoPerf Orchestrator..."

export PYTHONPATH="$(cd "$(dirname "$0")"/.. && pwd):${PYTHONPATH:-}"
echo "PYTHONPATH set to: $PYTHONPATH"

PYTHON_SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/src/main.py"
echo "Executing Python script: $PYTHON_SCRIPT_PATH with arguments: $@"

python "$PYTHON_SCRIPT_PATH" "$@"
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "AutoPerf Orchestrator finished successfully."
else
    echo "AutoPerf Orchestrator failed with exit code: $PYTHON_EXIT_CODE" >&2
    exit $PYTHON_EXIT_CODE
fi