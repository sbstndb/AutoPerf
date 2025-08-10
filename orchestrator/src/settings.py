"""
Configuration centralisée pour AutoPerf avec validation et support des variables d'environnement.
"""
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field, validator


KernelType = Literal["matvec", "matmul", "reduce", "search", "custom", "axpy"]


class PathSettings(BaseSettings):
    """Configuration des chemins du projet"""
    
    # Chemin racine du projet (calculé automatiquement)
    repo_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.absolute())
    
    @property
    def cpp_root(self) -> Path:
        return self.repo_root / "cpp"
    
    @property
    def build_dir(self) -> Path:
        return self.repo_root / "build"
    
    @property
    def runs_dir(self) -> Path:
        return self.repo_root / "orchestrator" / "runs"
    
    @property
    def best_dir(self) -> Path:
        return self.repo_root / "orchestrator" / "best"
    
    @property
    def prompts_dir(self) -> Path:
        return self.repo_root / "orchestrator" / "prompts"
    
    @property
    def env_path(self) -> Path:
        return self.repo_root / "orchestrator" / ".env"
    
    @property
    def orch_src_dir(self) -> Path:
        return self.repo_root / "orchestrator" / "src"


class KernelSettings(BaseSettings):
    """Configuration des kernels"""
    
    # Types de kernels supportés
    kernel_types: List[KernelType] = [
        "matvec", "matmul", "reduce", "search", "custom", "axpy"
    ]
    
    # Mapping des types vers les fichiers
    kernel_basenames: Dict[KernelType, str] = {
        "matvec": "kernel_matvec.cpp",
        "matmul": "kernel_matmul.cpp", 
        "reduce": "kernel_reduce.cpp",
        "search": "kernel_search.cpp",
        "custom": "kernel_custom.cpp",
        "axpy": "kernel_axpy.cpp",
    }
    
    def get_kernel_file_path(self, kernel_type: KernelType, cpp_root: Path) -> Path:
        """Retourne le chemin vers le fichier du kernel"""
        if kernel_type not in self.kernel_types:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        basename = self.kernel_basenames[kernel_type]
        return cpp_root / "src" / basename


class LLMSettings(BaseSettings):
    """Configuration des modèles LLM"""
    
    # Modèle par défaut
    default_model: str = Field(
        default="mistralai/mistral-7b-instruct",
        env="AUTOPERF_DEFAULT_MODEL",
        description="Modèle LLM par défaut"
    )
    
    # Température par défaut
    default_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        env="AUTOPERF_DEFAULT_TEMPERATURE",
        description="Température par défaut pour le LLM"
    )
    
    # Configuration du système de thinking
    thinking_mode: str = Field(
        default="disabled",
        env="AUTOPERF_THINKING_MODE",
        description="Mode de thinking par défaut (disabled, dynamic, budget)"
    )
    
    thinking_budget: int = Field(
        default=500,
        ge=0,
        le=2000,
        env="AUTOPERF_THINKING_BUDGET",
        description="Budget de tokens par défaut pour le thinking mode"
    )
    
    # Modèles supportés (peut être étendu)
    supported_models: List[str] = [
        "mistralai/mistral-7b-instruct",
        "meta-llama/llama-2-7b-chat",
        "gpt-3.5-turbo",
        "gpt-5-nano",
    ]
    
    @validator('default_model')
    def validate_model(cls, v, values):
        # Note: On ne valide pas strictement car de nouveaux modèles peuvent être ajoutés
        return v


class OptimizationSettings(BaseSettings):
    """Configuration de l'optimisation hiérarchique"""
    
    # Paramètres par défaut pour la recherche hiérarchique
    default_phases: int = Field(
        default=3,
        ge=1,
        le=10,
        env="AUTOPERF_DEFAULT_PHASES",
        description="Nombre de phases par défaut"
    )
    
    default_branching: int = Field(
        default=4,
        ge=1,
        le=20,
        env="AUTOPERF_DEFAULT_BRANCHING", 
        description="Facteur de branchement par défaut"
    )
    
    default_jobs: int = Field(
        default=1,
        ge=1,
        le=16,
        env="AUTOPERF_DEFAULT_JOBS",
        description="Nombre de jobs parallèles par défaut"
    )
    
    # Limites maximales
    max_phases: int = Field(default=10, description="Nombre maximum de phases")
    max_branching: int = Field(default=20, description="Facteur de branchement maximum")
    max_jobs: int = Field(default=16, description="Nombre maximum de jobs parallèles")
    
    # Prompts par défaut pour la recherche hiérarchique
    default_prompts: List[str] = [
        "algorithmic.txt",
        "parallelization.txt", 
        "vectorization.txt"
    ]


class BuildSettings(BaseSettings):
    """Configuration de la compilation et des tests"""
    
    # Options de compilation par défaut
    enable_benchmarks: bool = Field(
        default=True,
        env="AUTOPERF_ENABLE_BENCHMARKS",
        description="Activer les benchmarks par défaut"
    )
    
    enable_tests: bool = Field(
        default=True,
        env="AUTOPERF_ENABLE_TESTS", 
        description="Activer les tests par défaut"
    )
    
    # Build type par défaut
    build_type: str = Field(
        default="Release",
        env="AUTOPERF_BUILD_TYPE",
        description="Type de build par défaut (Release/Debug)"
    )
    
    # Parallélisme de compilation
    parallel_jobs: int = Field(
        default=8,
        ge=1,
        le=32,
        env="AUTOPERF_PARALLEL_JOBS",
        description="Nombre de threads pour la compilation en parallèle"
    )
    
    @validator('build_type')
    def validate_build_type(cls, v):
        if v not in ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]:
            raise ValueError(f"Invalid build type: {v}")
        return v


class LoggingSettings(BaseSettings):
    """Configuration du logging"""
    
    log_level: str = Field(
        default="INFO",
        env="AUTOPERF_LOG_LEVEL",
        description="Niveau de log"
    )
    
    log_file: Optional[str] = Field(
        default=None,
        env="AUTOPERF_LOG_FILE",
        description="Fichier de log (None = console uniquement)"
    )
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class AutoPerfSettings(BaseSettings):
    """Configuration principale d'AutoPerf"""
    
    # Sous-configurations
    paths: PathSettings = Field(default_factory=PathSettings)
    kernels: KernelSettings = Field(default_factory=KernelSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings)
    build: BuildSettings = Field(default_factory=BuildSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    
    # Configuration globale
    debug_mode: bool = Field(
        default=False,
        env="AUTOPERF_DEBUG",
        description="Mode debug"
    )
    
    class Config:
        env_prefix = "AUTOPERF_"
        case_sensitive = False
        env_nested_delimiter = "__"
        
    def get_kernel_file_path(self, kernel_type: KernelType) -> Path:
        """Retourne le chemin vers le fichier du kernel"""
        return self.kernels.get_kernel_file_path(kernel_type, self.paths.cpp_root)
    
    def get_default_prompt_paths(self) -> List[Path]:
        """Retourne la liste des chemins vers les prompts par défaut"""
        return [
            self.paths.prompts_dir / prompt_name 
            for prompt_name in self.optimization.default_prompts
        ]


# Instance globale de configuration (singleton pattern)
_settings_instance: Optional[AutoPerfSettings] = None


def get_settings() -> AutoPerfSettings:
    """Retourne l'instance globale de configuration"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AutoPerfSettings()
    return _settings_instance


def reload_settings() -> AutoPerfSettings:
    """Recharge la configuration (utile pour les tests)"""
    global _settings_instance
    _settings_instance = AutoPerfSettings()
    return _settings_instance


# Rétrocompatibilité avec l'ancienne config (à supprimer progressivement)
def get_legacy_config():
    """Retourne la configuration dans le format legacy pour la transition"""
    settings = get_settings()
    return {
        'REPO_ROOT': str(settings.paths.repo_root),
        'CPP_ROOT': str(settings.paths.cpp_root),
        'BUILD_DIR': str(settings.paths.build_dir),
        'RUNS_DIR': str(settings.paths.runs_dir),
        'BEST_DIR': str(settings.paths.best_dir),
        'PROMPTS_DIR': str(settings.paths.prompts_dir),
        'ENV_PATH': str(settings.paths.env_path),
        'ORCH_SRC_DIR': str(settings.paths.orch_src_dir),
        'KERNEL_TYPES': settings.kernels.kernel_types,
        'KERNEL_BASENAMES': settings.kernels.kernel_basenames,
    }
