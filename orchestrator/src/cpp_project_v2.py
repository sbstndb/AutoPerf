"""
Version refactorisée de CppProject utilisant la séparation des responsabilités.
"""
import os
import subprocess
from pathlib import Path
from typing import Optional

from orchestrator.src.simple_config import get_config
from orchestrator.src.command_runner import TimedCommandRunner
from orchestrator.src.cpp_builder import CppBuilder
from orchestrator.src.cpp_tester import CppTester
from orchestrator.src.cpp_benchmarker import CppBenchmarker
from orchestrator.src.kernel_manager import SafeKernelManager
from orchestrator.src.build_interfaces import BuildResult, TestResult, BenchmarkResult, ConfigureResult


class CppProject:
    """
    Gestionnaire de projet C++ refactorisé avec séparation des responsabilités.
    
    Cette classe orchestre les différents composants :
    - CppBuilder : Configuration et compilation
    - CppTester : Exécution des tests
    - CppBenchmarker : Exécution des benchmarks
    - KernelManager : Gestion du code source des kernels
    """
    
    def __init__(self, path: str, kernel_type: str):
        """
        Initialise un projet C++ pour un kernel spécifique
        
        Args:
            path: Chemin vers le répertoire du projet
            kernel_type: Type de kernel (matvec, matmul, etc.)
        """
        self.path = Path(path)
        self.kernel_type = kernel_type
        self.settings = get_config()
        
        # Valider le type de kernel
        if kernel_type not in self.settings.kernel_types:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        # Créer le répertoire du projet
        self.path.mkdir(parents=True, exist_ok=True)
        
        # Copier le projet C++ si nécessaire
        self._setup_project()
        
        # Initialiser les chemins
        self._source_dir = self.path
        self._build_dir = self.path / "build"
        
        # Initialiser les composants
        self.command_runner = TimedCommandRunner(default_cwd=self.path)
        self.kernel_manager = SafeKernelManager(self._source_dir, kernel_type)
        self.builder = CppBuilder(self._source_dir, self._build_dir, kernel_type, self.command_runner)
        self.tester = CppTester(self._build_dir, kernel_type, self.command_runner)
        self.benchmarker = CppBenchmarker(self._build_dir, kernel_type, self.command_runner)
    
    def _setup_project(self) -> None:
        """
        Configure le projet en copiant les fichiers source si nécessaire
        """
        cmake_file = self.path / "CMakeLists.txt"
        
        # Copier le projet seulement s'il n'existe pas ou est vide
        if not cmake_file.exists():
            cpp_root = self.settings.cpp_root
            try:
                # Copier tout le contenu du répertoire cpp
                subprocess.run(
                    ['cp', '-r', f'{cpp_root}/.', str(self.path)], 
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to copy C++ project: {e}")
    
    # Méthodes du KernelManager
    def update_kernel(self, kernel_code: str) -> bool:
        """
        Met à jour le code du kernel
        
        Args:
            kernel_code: Nouveau code C++ du kernel
            
        Returns:
            True si la mise à jour a réussi
        """
        return self.kernel_manager.update_kernel(kernel_code)
    
    def get_kernel_info(self) -> dict:
        """Retourne des informations sur le kernel"""
        return self.kernel_manager.get_kernel_info()
    
    # Méthodes du Builder
    def configure(self, enable_benches: bool = True, enable_tests: bool = True) -> tuple[bool, str]:
        """
        Configure le projet CMake
        
        Args:
            enable_benches: Activer les benchmarks
            enable_tests: Activer les tests
            
        Returns:
            Tuple (success, log) pour compatibilité avec l'ancienne API
        """
        try:
            result = self.builder.configure(enable_benches, enable_tests)
            return result.success, result.log
        except Exception as e:
            return False, str(e)
    
    def build(self, target: Optional[str] = None) -> tuple[bool, str]:
        """
        Build le projet
        
        Args:
            target: Target spécifique à builder (optionnel)
            
        Returns:
            Tuple (success, log) pour compatibilité avec l'ancienne API
        """
        try:
            result = self.builder.build(target)
            return result.success, result.log
        except Exception as e:
            return False, str(e)
    
    # Méthodes du Tester
    def test(self) -> tuple[bool, str]:
        """
        Exécute les tests
        
        Returns:
            Tuple (success, log) pour compatibilité avec l'ancienne API
        """
        try:
            result = self.tester.run_kernel_tests()
            return result.success, result.log
        except Exception as e:
            return False, str(e)
    
    # Méthodes du Benchmarker
    def benchmark(self) -> tuple[bool, str, Optional[float]]:
        """
        Exécute les benchmarks
        
        Returns:
            Tuple (success, log, score) pour compatibilité avec l'ancienne API
        """
        try:
            result = self.benchmarker.run_kernel_benchmark()
            return result.success, result.log, result.score
        except Exception as e:
            return False, str(e), None
    
    # Nouvelles méthodes avec les types de résultats modernes
    def configure_v2(self, enable_benches: bool = True, enable_tests: bool = True) -> ConfigureResult:
        """Version moderne de configure retournant ConfigureResult"""
        return self.builder.configure(enable_benches, enable_tests)
    
    def build_v2(self, target: Optional[str] = None) -> BuildResult:
        """Version moderne de build retournant BuildResult"""
        return self.builder.build(target)
    
    def test_v2(self) -> TestResult:
        """Version moderne de test retournant TestResult"""
        return self.tester.run_kernel_tests()
    
    def benchmark_v2(self) -> BenchmarkResult:
        """Version moderne de benchmark retournant BenchmarkResult"""
        return self.benchmarker.run_kernel_benchmark()
    
    # Méthodes utilitaires
    def clean(self) -> bool:
        """Nettoie les fichiers de build"""
        return self.builder.clean()
    
    def is_configured(self) -> bool:
        """Vérifie si le projet est configuré"""
        return self.builder.is_configured()
    
    def is_built(self) -> bool:
        """Vérifie si le projet a été buildé"""
        return self.builder.is_built()
    
    def get_build_status(self) -> dict:
        """
        Retourne le statut complet du build
        
        Returns:
            Dictionnaire avec les informations de statut
        """
        return {
            "configured": self.is_configured(),
            "built": self.is_built(),
            "kernel_info": self.get_kernel_info(),
            "available_tests": self.tester.list_available_tests(),
            "available_benchmarks": self.benchmarker.list_available_benchmarks(),
        }
    
    # Méthodes avancées
    def full_pipeline(self, kernel_code: str, enable_benches: bool = True, enable_tests: bool = True) -> dict:
        """
        Exécute le pipeline complet : update -> configure -> build -> test -> benchmark
        
        Args:
            kernel_code: Code du kernel à tester
            enable_benches: Activer les benchmarks
            enable_tests: Activer les tests
            
        Returns:
            Dictionnaire avec tous les résultats
        """
        results = {
            "kernel_update": False,
            "configure": None,
            "build": None,
            "test": None,
            "benchmark": None,
            "success": False
        }
        
        # 1. Mettre à jour le kernel
        if not self.update_kernel(kernel_code):
            results["error"] = "Failed to update kernel"
            return results
        results["kernel_update"] = True
        
        try:
            # 2. Configurer
            results["configure"] = self.configure_v2(enable_benches, enable_tests)
            if not results["configure"].success:
                return results
            
            # 3. Builder
            results["build"] = self.build_v2()
            if not results["build"].success:
                return results
            
            # 4. Tester
            if enable_tests:
                results["test"] = self.test_v2()
                if not results["test"].success:
                    return results
            
            # 5. Benchmarker
            if enable_benches:
                results["benchmark"] = self.benchmark_v2()
                if not results["benchmark"].success:
                    return results
            
            results["success"] = True
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def performance_analysis(self, kernel_code: str, iterations: int = 5) -> dict:
        """
        Analyse de performance complète avec multiple itérations
        
        Args:
            kernel_code: Code du kernel
            iterations: Nombre d'itérations pour les benchmarks
            
        Returns:
            Dictionnaire avec l'analyse complète
        """
        # Mettre à jour et builder
        pipeline_result = self.full_pipeline(kernel_code)
        if not pipeline_result["success"]:
            return pipeline_result
        
        # Analyse de performance détaillée
        try:
            benchmark_result = self.benchmarker.run_multiple_iterations(iterations)
            
            return {
                "success": True,
                "pipeline": pipeline_result,
                "performance": {
                    "score": benchmark_result.score,
                    "measurements": benchmark_result.measurements,
                    "log": benchmark_result.log,
                    "iterations": iterations
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "pipeline": pipeline_result,
                "performance_error": str(e)
            }


# Alias pour compatibilité avec l'ancienne API
CppProjectV1 = CppProject  # L'ancienne version sera dans cpp_project.py


class OptimizedCppProject(CppProject):
    """Version optimisée avec cache et builds incrémentaux"""
    
    def __init__(self, path: str, kernel_type: str):
        super().__init__(path, kernel_type)
        
        # Utiliser les versions optimisées des composants
        from orchestrator.src.cpp_builder import OptimizedCppBuilder
        self.builder = OptimizedCppBuilder(
            self._source_dir, self._build_dir, kernel_type, self.command_runner
        )
    
    def smart_rebuild(self, kernel_code: str) -> dict:
        """
        Rebuild intelligent qui ne recompile que si nécessaire
        
        Args:
            kernel_code: Code du kernel
            
        Returns:
            Dictionnaire avec les résultats
        """
        results = {"changes_detected": False, "rebuild_needed": False}
        
        # Vérifier si le code a changé
        current_code = self.kernel_manager.read_kernel_code()
        if current_code == kernel_code:
            results["message"] = "No changes detected, skipping rebuild"
            results["success"] = True
            return results
        
        results["changes_detected"] = True
        results["rebuild_needed"] = True
        
        # Effectuer le rebuild
        return self.full_pipeline(kernel_code)