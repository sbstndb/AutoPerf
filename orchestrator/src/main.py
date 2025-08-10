#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import importlib

def _init_langchain_globals():
    try:
        lc_globals = importlib.import_module("langchain.globals")
        return lc_globals.set_verbose, lc_globals.set_debug
    except (ImportError, ModuleNotFoundError):
        return (lambda _: None), (lambda _: None)

set_verbose, set_debug = _init_langchain_globals()

from orchestrator.src.config import (
    ENV_PATH,
    KERNEL_TYPES,
    PROMPTS_DIR,
    kernel_file_path,
)
from orchestrator.src.cpp_project import CppProject
from orchestrator.src.langchain_optimizer import create_langchain_optimizer
from orchestrator.src.optimizer import Optimizer
from orchestrator.src.types import Baseline, OptResult
from orchestrator.src.ui import UI
from orchestrator.src.utils import load_env_file
from orchestrator.src.workspace import Workspace
from orchestrator.src.errors import (
    ConfigureError,
    BuildError,
    TestError,
    BenchmarkError,
    BenchmarkParseError,
    OperationTimeout,
)

set_verbose(False)
set_debug(False)


def measure_baseline(ui, workspace, kernel_type, kernel_file):
    """Measure the performance of the initial C++ kernel."""
    baseline_cpp_path = workspace.get_cpp_project_path(0)  # Using 0 for baseline iteration
    project = CppProject(baseline_cpp_path, kernel_type)
    with open(kernel_file, 'r', encoding='utf-8') as f:
        kernel_code = f.read()
        project.update_kernel(kernel_code)

        log = ""
        try:
            _, configure_log = project.configure()
            log += configure_log
        except (ConfigureError, OperationTimeout) as e:
            log += str(e)
            return OptResult(
                0,
                parent_id=None,
                success=False,
                compile_ok=False,
                tests_ok=False,
                benchmark_ns_per_item=None,
                kernel_code=kernel_code,
                log=log,
                failure_stage="configure",
            )

        try:
            _, build_log = project.build()
            log += build_log
        except (BuildError, OperationTimeout) as e:
            log += str(e)
            return OptResult(
                0,
                parent_id=None,
                success=False,
                compile_ok=False,
                tests_ok=False,
                benchmark_ns_per_item=None,
                kernel_code=kernel_code,
                log=log,
                failure_stage="compile",
            )

        try:
            _, test_log = project.test()
            log += test_log
        except (TestError, OperationTimeout) as e:
            log += str(e)
            return OptResult(
                0,
                parent_id=None,
                success=False,
                compile_ok=True,
                tests_ok=False,
                benchmark_ns_per_item=None,
                kernel_code=kernel_code,
                log=log,
                failure_stage="tests",
            )

        try:
            _, bench_log, score = project.benchmark()
            log += bench_log
        except (BenchmarkParseError, BenchmarkError, OperationTimeout) as e:
            log += str(e)
            ui.console.print(f"[red]Benchmark failed: {e}[/red]")
            return OptResult(
                0,
                parent_id=None,
                success=False,
                compile_ok=True,
                tests_ok=True,
                benchmark_ns_per_item=None,
                kernel_code=kernel_code,
                log=log,
                failure_stage="bench_fail",
            )

        return OptResult(
            0,
            parent_id=None,
            success=True,
            compile_ok=True,
            tests_ok=True,
            benchmark_ns_per_item=score,
            kernel_code=kernel_code,
            log=log,
        )


def process_llm_and_build(
    workspace,
    kernel_type,
    parent_code,
    parent_score,
    phase,
    ui,
    iteration,
    parent_id,
    # Langchain config
    model,
    temperature,
    prompt_file,
    prompt_files,
    thinking_config=None,
):
    langchain_optimizer = create_langchain_optimizer(
        model=model,
        temperature=temperature,
        prompt_file=prompt_file,
        prompt_files=prompt_files,
        thinking_config=thinking_config,
    )

    ui.update_iteration_status(iteration, "LLM", "🔄")
    try:
        new_kernel_code = langchain_optimizer.invoke(
            {"kernel_code": parent_code, "score": parent_score, "phase": phase - 1}
        )
        ui.update_iteration_status(iteration, "LLM", "✅")
    except Exception as e:  # noqa: BLE001  # pylint: disable=broad-except
        ui.update_iteration_status(iteration, "LLM", "❌")
        return {
            "success": False, 
            "result": None, 
            "project_path": None, 
            "iteration": iteration,
            "parent_id": parent_id,
            "kernel_code": parent_code,
            "log": f"LLM failed: {e}",
            "failure_stage": "llm"
        }

    cpp_project_path = workspace.get_cpp_project_path(iteration)
    project = CppProject(cpp_project_path, kernel_type)
    project.update_kernel(new_kernel_code)

    log = ""

    ui.update_iteration_status(iteration, "Configure", "🔄")
    try:
        _, configure_log = project.configure()
        log += configure_log
        ui.update_iteration_status(iteration, "Configure", "✅")
    except (ConfigureError, OperationTimeout) as e:
        log += str(e)
        ui.update_iteration_status(iteration, "Configure", "❌")
        return {
            "success": False,
            "iteration": iteration,
            "parent_id": parent_id,
            "kernel_code": new_kernel_code,
            "log": log,
            "failure_stage": "configure",
        }

    ui.update_iteration_status(iteration, "Build", "🔄")
    try:
        _, build_log = project.build()
        log += build_log
        ui.update_iteration_status(iteration, "Build", "✅")
    except (BuildError, OperationTimeout) as e:
        log += str(e)
        ui.update_iteration_status(iteration, "Build", "❌")
        return {
            "success": False,
            "iteration": iteration,
            "parent_id": parent_id,
            "kernel_code": new_kernel_code,
            "log": log,
            "failure_stage": "compile",
        }

    return {
        "success": True,
        "iteration": iteration,
        "parent_id": parent_id,
        "kernel_code": new_kernel_code,
        "log": log,
        "project_path": str(cpp_project_path),
    }


def parse_args():
    """Parse command line arguments for the orchestrator."""
    parser = argparse.ArgumentParser(
        description="AutoPerf: Automatic Performance Optimization for C++ Kernels"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="matvec",
        choices=KERNEL_TYPES,
        help="Kernel to optimize.",
    )
    parser.add_argument(
        "--model", type=str, default="mistralai/mistral-7b-instruct", help="Default OpenAI model."
    )
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--prompt", type=str, help="Path to a custom prompt file.")
    parser.add_argument(
        "--prompts",
        type=str,
        help="Comma-separated paths to custom prompt files for hierarchical search.",
    )
    parser.add_argument("--parent-mode", type=str, default="best", choices=["best", "random_success"])
    parser.add_argument("--phases", type=int, default=3, help="Depth of the hierarchical search tree.")
    parser.add_argument("--branching", type=int, default=4, help="Breadth (branching factor) per phase.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs for LLM and build stages.")
    
    # Thinking system parameters
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--thinking-disabled", 
        action="store_true", 
        help="Disable thinking mode completely (default behavior)"
    )
    thinking_group.add_argument(
        "--thinking-dynamic", 
        action="store_true", 
        help="Enable dynamic thinking mode (LLM decides when to think)"
    )
    thinking_group.add_argument(
        "--thinking-budget", 
        type=int, 
        metavar="TOKENS",
        help="Enable thinking mode with fixed token budget (e.g., --thinking-budget 500)"
    )
    
    return parser.parse_args()


def resolve_kernel_file(kernel):
    """Return the source file path for the selected kernel type."""
    return kernel_file_path(kernel)


def prepare_prompts(args):
    """Return list of prompt files to use for the current run."""
    if args.prompts:
        return args.prompts.split(",")
    return [
        os.path.join(PROMPTS_DIR, "algorithmic.txt"),
        os.path.join(PROMPTS_DIR, "parallelization.txt"),
        os.path.join(PROMPTS_DIR, "vectorization.txt"),
    ]


def _result_from_failure_dict(res):
    """Convert a failure dict from process_llm_and_build into an OptResult."""
    iteration_id = res.get("iteration", 0)
    parent_id = res.get("parent_id", 0)
    kernel_code = res.get("kernel_code", "")
    log = res.get("log", "")
    failure_stage = res.get("failure_stage", "unknown")
    return OptResult(
        id=iteration_id,
        parent_id=parent_id,
        success=False,
        compile_ok=False,
        tests_ok=False,
        benchmark_ns_per_item=None,
        kernel_code=kernel_code,
        log=log,
        failure_stage=failure_stage,
    )


def run_hierarchical_search(args, ui, workspace, optimizer, baseline, thinking_config=None):
    """Run the hierarchical multi-phase search strategy."""
    iter_counter = 1
    frontier = [0]  # start from baseline id

    for phase in range(1, args.phases + 1):
        # Phase start (tree UI handles the visualization)

        tasks = []
        for parent_id in frontier:
            parent_result = optimizer.get_result_by_id(parent_id)
            for _ in range(args.branching):
                tasks.append((parent_result.kernel_code, parent_result.score, iter_counter, parent_id))
                iter_counter += 1

        ui.start_parallel_iterations([task[2] for task in tasks])

        for _, _, it_id, p_id in tasks:
            ui.add_iteration_to_tree(it_id, p_id)

        built_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_task = {
                executor.submit(
                    process_llm_and_build,
                    workspace,
                    args.kernel,
                    p_code,
                    p_score,
                    phase,
                    ui,
                    it_id,
                    p_id,
                    args.model,
                    args.temperature,
                    args.prompt if args.prompt else None,
                    prepare_prompts(args),
                    thinking_config,
                ): (p_code, p_score, it_id, p_id)
                for p_code, p_score, it_id, p_id in tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                res = future.result()
                built_results.append(res)

        # Sequential testing phase (kept implicit, UI remains active)

        next_frontier_ids = []
        for res in built_results:
            if not res["success"]:
                result = _result_from_failure_dict(res)
                optimizer.add_result(result)
                workspace.save_iteration(result.id, result)
                ui.update_iteration_performance(result.id, None)
                # Tree UI updates already convey the result status
                continue

            project = CppProject(res["project_path"], args.kernel)
            log = res["log"]

            # Tests stage with unified exceptions
            try:
                _, test_log = project.test()
                log += test_log
            except (TestError, OperationTimeout) as e:
                log += str(e)
                result = OptResult(
                    id=res["iteration"],
                    parent_id=res["parent_id"],
                    success=False,
                    compile_ok=True,
                    tests_ok=False,
                    benchmark_ns_per_item=None,
                    kernel_code=res["kernel_code"],
                    log=log,
                    failure_stage="tests",
                )
                optimizer.add_result(result)
                workspace.save_iteration(res["iteration"], result)
                ui.update_iteration_performance(res["iteration"], None)
                continue

            # Benchmark stage with unified exceptions
            try:
                _, bench_log, score = project.benchmark()
                log += bench_log
                result = OptResult(
                    id=res["iteration"],
                    parent_id=res["parent_id"],
                    success=True,
                    compile_ok=True,
                    tests_ok=True,
                    benchmark_ns_per_item=score,
                    kernel_code=res["kernel_code"],
                    log=log,
                )
                ui.update_iteration_performance(res["iteration"], score, baseline.score)
                next_frontier_ids.append(result.id)
            except (BenchmarkParseError, BenchmarkError, OperationTimeout) as e:
                log += str(e)
                result = OptResult(
                    id=res["iteration"],
                    parent_id=res["parent_id"],
                    success=False,
                    compile_ok=True,
                    tests_ok=True,
                    benchmark_ns_per_item=None,
                    kernel_code=res["kernel_code"],
                    log=log,
                    failure_stage="bench_fail",
                )
                ui.update_iteration_performance(res["iteration"], None)

            optimizer.add_result(result)
            workspace.save_iteration(res["iteration"], result)
            # Tree UI updates already convey the result status

        if next_frontier_ids:
            frontier = next_frontier_ids


def run_pipeline():
    """Entry point to run the orchestrator pipeline with refactored steps."""
    args = parse_args()
    load_env_file(ENV_PATH)
    
    # Créer la configuration de thinking à partir des arguments CLI
    from orchestrator.src.thinking_config import ThinkingConfig
    thinking_config = ThinkingConfig.from_cli_args(args)

    ui = UI()
    workspace = Workspace(kernel_type=args.kernel)
    ui.start(str(workspace.path))

    kernel_file = resolve_kernel_file(args.kernel)

    baseline_result = measure_baseline(ui, workspace, args.kernel, kernel_file)
    if not baseline_result or not baseline_result.success:
        sys.exit(1)

    baseline = Baseline(success=True, score=baseline_result.score, log=baseline_result.log)
    ui.print_baseline(baseline)
    workspace.save_baseline_log(baseline.log)

    optimizer = Optimizer(args.parent_mode, baseline_result.kernel_code, baseline_result.score)
    optimizer.add_result(baseline_result)

    # Hierarchical search is the only supported mode
    run_hierarchical_search(args, ui, workspace, optimizer, baseline, thinking_config)

    if optimizer.best_result:
        workspace.save_best_kernel(optimizer.best_result.kernel_code)

    ui.done(optimizer.best_result, optimizer.history, args.kernel)


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
