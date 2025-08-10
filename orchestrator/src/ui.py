import threading

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.tree import Tree

# Note: Avoid importing types here to keep UI decoupled and reduce linter noise


class UI:
    """Handles all console output for the optimization process."""

    def __init__(self):
        self.console = Console()
        self.lock = threading.Lock()
        self.table = None
        self.live_display = None
        self.iterations_status = {}
        self.parallel_mode = False
        self.optimization_tree = {}
        self.baseline_score = None

    def start(self, run_dir: str):
        self.console.rule(f"AutoPerf Run: {run_dir}")
        self.table = Table(show_header=True, header_style="bold magenta")
        self.table.add_column("Iter")
        self.table.add_column("Score")
        self.table.add_column("vs Best")
        self.table.add_column("vs Base")
        self.table.add_column("Result")

    def start_parallel_iterations(self, iterations):
        """Starts real-time display for parallel iterations"""
        self.parallel_mode = True
        
        # Add new iterations to existing status
        for i in iterations:
            if i not in self.iterations_status:
                self.iterations_status[i] = {"LLM": "⏳", "Configure": "⏳", "Build": "⏳", "Performance": "⏳"}
        
        # Create real-time tree
        tree = self._create_optimization_tree()
        
        if self.live_display is None:
            self.live_display = Live(tree, refresh_per_second=4, console=self.console)
            self.live_display.start()
        else:
            self.live_display.update(tree)

    def _create_optimization_tree(self):
        """Creates the optimization tree for display"""
        tree = Tree("🚀 AutoPerf Optimization Tree", style="bold blue")
        
        # Add baseline if available
        if self.baseline_score is not None:
            baseline_node = tree.add(f"📊 Baseline ({self.baseline_score:.1f} ns) ✅", style="bold green")
        else:
            baseline_node = tree.add("📊 Baseline ⏳", style="yellow")
        
        # Add iterations
        for iter_num in sorted(self.iterations_status.keys()):
            statuses = self.iterations_status[iter_num]
            parent_id = self.optimization_tree.get(iter_num, {}).get('parent_id', 0)
            
            # Find parent node
            parent_node = baseline_node if parent_id == 0 else self._find_node_by_iter(tree, parent_id)
            if parent_node is None:
                parent_node = baseline_node
            
            # Create iteration node
            if statuses["Performance"] != "⏳":
                parent_node.add(f"🔄 Iter {iter_num} ({statuses['Performance']})", style="cyan")
            else:
                status_str = f"LLM:{statuses['LLM']} Config:{statuses['Configure']} Build:{statuses['Build']}"
                parent_node.add(f"🔄 Iter {iter_num} [{status_str}]", style="yellow")
        
        return tree

    def _find_node_by_iter(self, tree, iter_num):
        """Recursively finds a node by iteration number"""
        def search_node(node, target_iter):
            if f"Iter {target_iter}" in str(node.label):
                return node
            for child in node.children:
                result = search_node(child, target_iter)
                if result:
                    return result
            return None
        
        return search_node(tree, iter_num)

    def update_iteration_status(self, iteration, stage, status):
        """Updates the status of a specific stage"""
        with self.lock:
            if self.parallel_mode and self.live_display:
                if iteration in self.iterations_status:
                    self.iterations_status[iteration][stage] = status
                    
                    # Recreate tree with updated status
                    tree = self._create_optimization_tree()
                    self.live_display.update(tree)

    def update_iteration_performance(self, iteration, score, baseline_score=None):
        """Updates the performance score for an iteration"""
        with self.lock:
            if self.parallel_mode and self.live_display:
                if iteration in self.iterations_status:
                    if score is not None:
                        if baseline_score is not None:
                            improvement = ((baseline_score - score) / baseline_score) * 100
                            if improvement > 0:
                                perf_str = f"{score:.1f} (+{improvement:.1f}%)"
                            else:
                                perf_str = f"{score:.1f} ({improvement:.1f}%)"
                        else:
                            perf_str = f"{score:.1f}"
                        self.iterations_status[iteration]["Performance"] = perf_str
                    else:
                        self.iterations_status[iteration]["Performance"] = "❌"
                    
                    # Recreate tree with updated status
                    tree = self._create_optimization_tree()
                    self.live_display.update(tree)

    def set_baseline_score(self, score):
        """Sets the baseline score for the tree display"""
        self.baseline_score = score

    def add_iteration_to_tree(self, iteration, parent_id=0):
        """Adds an iteration to the optimization tree"""
        self.optimization_tree[iteration] = {'parent_id': parent_id}

    def stop_parallel_display(self):
        """Stops the real-time display"""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None
            self.parallel_mode = False
            self.console.print()  # Add a newline after the live display

    def print_baseline(self, baseline):
        if baseline.success:
            self.set_baseline_score(baseline.score)
        else:
            self.console.print("[red]Baseline failed:[/red]")
            self.console.print(baseline.log)

    # Removed unused UI methods (phase prints, sequential testing, legacy printing)

    def done(self, best_result, history, kernel_type="matvec"):
        # Stop the dynamic display first
        if self.live_display:
            self.stop_parallel_display()
        
        # Final tree display
        self.console.rule("Final Optimization Tree")
        final_tree = self._create_optimization_tree()
        self.console.print(final_tree)

        self.console.rule("Done")
        _ = history  # intentionally unused; kept for API compatibility
        if best_result:
            from orchestrator.src.config import KERNEL_BASENAMES  # local import to avoid cycles
            file_name = KERNEL_BASENAMES.get(kernel_type, "kernel.cpp")
            self.console.print(
                Panel(
                    f"[bold green]Best score: {best_result.score:.2f} ns/item[/bold green]"
                    f"\nBest kernel saved to: orchestrator/best/{file_name}"
                )
            )
        else:
            self.console.print(Panel("[yellow]No successful optimizations.[/yellow]"))

