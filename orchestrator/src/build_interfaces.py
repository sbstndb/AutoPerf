"""
Interfaces et types pour les composants de build, test et benchmark.
"""
from abc import ABC, abstractmethod
from typing import Protocol, Tuple, Optional, NamedTuple
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class BuildResult:
    """Résultat d'une opération de build"""
    success: bool
    log: str
    build_time: Optional[float] = None
    
    
@dataclass(frozen=True)
class TestResult:
    """Résultat d'une opération de test"""
    success: bool
    log: str
    tests_passed: Optional[int] = None
    tests_failed: Optional[int] = None
    test_time: Optional[float] = None
    

@dataclass(frozen=True)
class BenchmarkResult:
    """Résultat d'une opération de benchmark"""
    success: bool
    log: str
    score: Optional[float] = None
    benchmark_time: Optional[float] = None
    measurements: Optional[list] = None


@dataclass(frozen=True)
class ConfigureResult:
    """Résultat d'une opération de configuration"""
    success: bool
    log: str
    configure_time: Optional[float] = None


class ICppBuilder(Protocol):
    """Interface pour les composants de build C++"""
    
    def configure(self, enable_benchmarks: bool = True, enable_tests: bool = True) -> ConfigureResult:
        """Configure le projet CMake"""
        ...
    
    def build(self, target: Optional[str] = None) -> BuildResult:
        """Build le projet ou une target spécifique"""
        ...
    
    def clean(self) -> bool:
        """Nettoie les fichiers de build"""
        ...


class ICppTester(Protocol):
    """Interface pour les composants de test C++"""
    
    def run_tests(self, test_pattern: Optional[str] = None) -> TestResult:
        """Exécute les tests"""
        ...
    
    def run_specific_test(self, test_name: str) -> TestResult:
        """Exécute un test spécifique"""
        ...


class ICppBenchmarker(Protocol):
    """Interface pour les composants de benchmark C++"""
    
    def run_benchmark(self, benchmark_pattern: Optional[str] = None) -> BenchmarkResult:
        """Exécute les benchmarks"""
        ...
    
    def run_specific_benchmark(self, benchmark_name: str) -> BenchmarkResult:
        """Exécute un benchmark spécifique"""
        ...


class IKernelManager(Protocol):
    """Interface pour la gestion des kernels"""
    
    def update_kernel(self, kernel_code: str) -> bool:
        """Met à jour le code du kernel"""
        ...
    
    def get_kernel_path(self) -> Path:
        """Retourne le chemin vers le fichier kernel"""
        ...


class ICommandRunner(Protocol):
    """Interface pour l'exécution de commandes"""
    
    def run_command(self, command: list[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
        """Exécute une commande et retourne (success, log)"""
        ...


# Types d'alias pour améliorer la lisibilité
KernelType = str
CommandList = list[str]
LogOutput = str