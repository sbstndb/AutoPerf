"""
Composant responsable de l'exécution et analyse des benchmarks C++.
"""
import re
import statistics
from pathlib import Path
from typing import Optional, List, Dict, Any
from orchestrator.src.build_interfaces import ICppBenchmarker, BenchmarkResult
from orchestrator.src.command_runner import TimedCommandRunner
from orchestrator.src.errors import BenchmarkError, BenchmarkParseError


class CppBenchmarker:
    """Gestionnaire des benchmarks pour les projets C++"""
    
    def __init__(
        self,
        build_dir: Path,
        kernel_type: str,
        command_runner: Optional[TimedCommandRunner] = None
    ):
        self.build_dir = Path(build_dir)
        self.kernel_type = kernel_type
        self.command_runner = command_runner or TimedCommandRunner(default_cwd=self.build_dir.parent)
    
    def run_benchmark(self, benchmark_pattern: Optional[str] = None) -> BenchmarkResult:
        """
        Exécute les benchmarks pour le kernel
        
        Args:
            benchmark_pattern: Pattern pour filtrer les benchmarks (optionnel)
            
        Returns:
            BenchmarkResult avec le score et les détails
            
        Raises:
            BenchmarkError: En cas d'échec du benchmark
            BenchmarkParseError: En cas d'échec du parsing des résultats
        """
        benchmark_executable = self._get_benchmark_executable()
        
        if not benchmark_executable.exists():
            raise BenchmarkError(f"Benchmark executable not found: {benchmark_executable}")
        
        # Construire la commande de benchmark
        command = [str(benchmark_executable)]
        
        # Ajouter des options de benchmark si nécessaire
        if benchmark_pattern:
            command.extend(["--benchmark_filter", benchmark_pattern])
        
        # Options pour un output plus stable
        command.extend([
            "--benchmark_repetitions=3",  # Répéter 3 fois pour plus de stabilité
            "--benchmark_display_aggregates_only=true",  # Afficher seulement les agrégats
            "--benchmark_format=console",  # Format console lisible
        ])
        
        try:
            success, log, benchmark_time = self.command_runner.run_command(command)
            
            if not success:
                raise BenchmarkError(f"Benchmark execution failed:\n{log}")
            
            # Parser les résultats du benchmark
            score, measurements = self._parse_benchmark_results(log)
            
            return BenchmarkResult(
                success=True,
                log=log,
                score=score,
                benchmark_time=benchmark_time,
                measurements=measurements
            )
            
        except Exception as e:
            if isinstance(e, (BenchmarkError, BenchmarkParseError)):
                raise
            raise BenchmarkError(f"Benchmark execution error: {e}") from e
    
    def run_specific_benchmark(self, benchmark_name: str) -> BenchmarkResult:
        """
        Exécute un benchmark spécifique
        
        Args:
            benchmark_name: Nom exact du benchmark
            
        Returns:
            BenchmarkResult
        """
        return self.run_benchmark(benchmark_pattern=f"^{re.escape(benchmark_name)}$")
    
    def run_kernel_benchmark(self) -> BenchmarkResult:
        """
        Exécute le benchmark du kernel actuel
        
        Returns:
            BenchmarkResult
        """
        return self.run_benchmark(benchmark_pattern=f"BM_{self.kernel_type}")
    
    def run_multiple_iterations(self, iterations: int = 5) -> BenchmarkResult:
        """
        Exécute le benchmark plusieurs fois pour plus de stabilité
        
        Args:
            iterations: Nombre d'itérations
            
        Returns:
            BenchmarkResult avec statistiques agrégées
        """
        results = []
        all_logs = []
        
        for i in range(iterations):
            try:
                result = self.run_kernel_benchmark()
                if result.success and result.score is not None:
                    results.append(result.score)
                    all_logs.append(f"Iteration {i+1}:\n{result.log}")
            except Exception as e:
                all_logs.append(f"Iteration {i+1} failed: {e}")
        
        if not results:
            raise BenchmarkError("All benchmark iterations failed")
        
        # Calculer les statistiques
        mean_score = statistics.mean(results)
        median_score = statistics.median(results)
        std_dev = statistics.stdev(results) if len(results) > 1 else 0.0
        
        combined_log = "\n".join(all_logs)
        combined_log += f"\n\nStatistics over {len(results)} successful runs:"
        combined_log += f"\nMean: {mean_score:.2f} ns"
        combined_log += f"\nMedian: {median_score:.2f} ns"
        combined_log += f"\nStd Dev: {std_dev:.2f} ns"
        combined_log += f"\nMin: {min(results):.2f} ns"
        combined_log += f"\nMax: {max(results):.2f} ns"
        
        return BenchmarkResult(
            success=True,
            log=combined_log,
            score=mean_score,  # Utiliser la moyenne comme score final
            measurements=results
        )
    
    def _get_benchmark_executable(self) -> Path:
        """
        Retourne le chemin vers l'exécutable de benchmark
        
        Returns:
            Path vers l'exécutable
        """
        return self.build_dir / "bin" / f"bench_{self.kernel_type}"
    
    def _parse_benchmark_results(self, log: str) -> tuple[float, List[float]]:
        """
        Parse les résultats du benchmark depuis la sortie
        
        Args:
            log: Sortie du benchmark
            
        Returns:
            Tuple (score_total, liste_des_mesures)
            
        Raises:
            BenchmarkParseError: Si le parsing échoue
        """
        kernel_name_for_regex = self.kernel_type
        
        # Patterns pour différents formats de benchmark
        patterns = [
            # Google Benchmark format standard
            rf"BM_{kernel_name_for_regex}(?:/\d+)*\s+(\d+\.?\d*)\s+ns",
            # Format avec taille
            rf"BM_{kernel_name_for_regex}/(\d+)\s+(\d+\.?\d*)\s+ns",
            # Format agrégé
            rf"BM_{kernel_name_for_regex}_mean\s+(\d+\.?\d*)\s+ns",
        ]
        
        measurements = []
        
        for pattern in patterns:
            matches = re.findall(pattern, log, re.IGNORECASE)
            if matches:
                # Extraire les scores (dernier groupe de chaque match)
                for match in matches:
                    if isinstance(match, tuple):
                        score = float(match[-1])  # Dernier élément du tuple
                    else:
                        score = float(match)
                    measurements.append(score)
        
        if not measurements:
            # Essayer des patterns plus génériques
            generic_patterns = [
                rf"{kernel_name_for_regex}.*?(\d+\.?\d*)\s*ns",
                r"(\d+\.?\d*)\s*ns/op",
                r"Time:\s*(\d+\.?\d*)\s*ns",
            ]
            
            for pattern in generic_patterns:
                matches = re.findall(pattern, log, re.IGNORECASE)
                if matches:
                    measurements = [float(match) for match in matches]
                    break
        
        if not measurements:
            raise BenchmarkParseError(
                f"Failed to parse benchmark scores for {kernel_name_for_regex} from output.\n"
                f"Output was:\n{log}"
            )
        
        # Calculer le score total (somme de toutes les mesures)
        total_score = sum(measurements)
        
        # Ajouter les informations de parsing au log
        parsing_info = f"\nFound {len(measurements)} benchmark measurements: {measurements}"
        parsing_info += f" -> Total: {total_score:.2f} ns"
        
        return total_score, measurements
    
    def list_available_benchmarks(self) -> List[str]:
        """
        Liste tous les benchmarks disponibles
        
        Returns:
            Liste des noms de benchmarks
        """
        benchmark_executable = self._get_benchmark_executable()
        
        if not benchmark_executable.exists():
            return []
        
        command = [str(benchmark_executable), "--benchmark_list_tests"]
        
        try:
            success, log, _ = self.command_runner.run_command(command)
            if success:
                # Parser la liste des benchmarks
                benchmarks = []
                for line in log.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        benchmarks.append(line)
                return benchmarks
        except Exception:
            pass
        
        return []
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        """
        Retourne des informations sur l'environnement de benchmark
        
        Returns:
            Dictionnaire avec les informations système
        """
        info = {
            "kernel_type": self.kernel_type,
            "executable_path": str(self._get_benchmark_executable()),
            "executable_exists": self._get_benchmark_executable().exists(),
        }
        
        # Essayer de récupérer des infos système du benchmark
        benchmark_executable = self._get_benchmark_executable()
        if benchmark_executable.exists():
            try:
                command = [str(benchmark_executable), "--benchmark_list_tests", "--benchmark_context"]
                success, log, _ = self.command_runner.run_command(command)
                if success:
                    info["system_info"] = log
            except Exception:
                pass
        
        return info


class AdvancedCppBenchmarker(CppBenchmarker):
    """Version avancée avec analyses statistiques poussées"""
    
    def run_performance_profile(self, warm_up_runs: int = 3, measurement_runs: int = 10) -> BenchmarkResult:
        """
        Exécute un profil de performance complet
        
        Args:
            warm_up_runs: Nombre de runs de warm-up
            measurement_runs: Nombre de runs de mesure
            
        Returns:
            BenchmarkResult avec analyse statistique complète
        """
        # Phase de warm-up
        for _ in range(warm_up_runs):
            try:
                self.run_kernel_benchmark()
            except Exception:
                pass  # Ignorer les erreurs de warm-up
        
        # Phase de mesure
        return self.run_multiple_iterations(measurement_runs)
    
    def compare_with_baseline(self, baseline_score: float) -> Dict[str, Any]:
        """
        Compare les performances avec un baseline
        
        Args:
            baseline_score: Score de référence
            
        Returns:
            Dictionnaire avec l'analyse comparative
        """
        result = self.run_kernel_benchmark()
        
        if not result.success or result.score is None:
            return {"error": "Benchmark failed"}
        
        improvement = ((baseline_score - result.score) / baseline_score) * 100
        
        return {
            "current_score": result.score,
            "baseline_score": baseline_score,
            "improvement_percent": improvement,
            "is_better": result.score < baseline_score,
            "speedup_factor": baseline_score / result.score if result.score > 0 else float('inf')
        }