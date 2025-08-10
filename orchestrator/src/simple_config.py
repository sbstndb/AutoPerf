"""
Configuration simplifiée pour AutoPerf sans dépendances externes.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional


class SimpleConfig:
    """Configuration simple sans validation externe"""
    
    def __init__(self):
        # Chemins du projet
        self.repo_root = Path(__file__).parent.parent.parent.absolute()
        self.cpp_root = self.repo_root / "cpp"
        self.build_dir = self.repo_root / "build"
        self.runs_dir = self.repo_root / "orchestrator" / "runs"
        self.best_dir = self.repo_root / "orchestrator" / "best"
        self.prompts_dir = self.repo_root / "orchestrator" / "prompts"
        self.env_path = self.repo_root / "orchestrator" / ".env"
        self.orch_src_dir = self.repo_root / "orchestrator" / "src"
        
        # Configuration des kernels
        self.kernel_types = ["matvec", "matmul", "reduce", "search", "custom", "axpy"]
        self.kernel_basenames = {
            "matvec": "kernel_matvec.cpp",
            "matmul": "kernel_matmul.cpp",
            "reduce": "kernel_reduce.cpp",
            "search": "kernel_search.cpp",
            "custom": "kernel_custom.cpp",
            "axpy": "kernel_axpy.cpp",
        }
        
        # Configuration LLM
        self.default_model = os.getenv("AUTOPERF_DEFAULT_MODEL", "mistralai/mistral-7b-instruct")
        self.default_temperature = float(os.getenv("AUTOPERF_DEFAULT_TEMPERATURE", "0.2"))
        
        # Configuration d'optimisation
        self.default_phases = int(os.getenv("AUTOPERF_DEFAULT_PHASES", "3"))
        self.default_branching = int(os.getenv("AUTOPERF_DEFAULT_BRANCHING", "4"))
        self.default_jobs = int(os.getenv("AUTOPERF_DEFAULT_JOBS", "1"))
        
        # Configuration de build
        self.enable_benchmarks = os.getenv("AUTOPERF_ENABLE_BENCHMARKS", "true").lower() == "true"
        self.enable_tests = os.getenv("AUTOPERF_ENABLE_TESTS", "true").lower() == "true"
        self.build_type = os.getenv("AUTOPERF_BUILD_TYPE", "Release")
        
        # Configuration de logging
        self.log_level = os.getenv("AUTOPERF_LOG_LEVEL", "INFO").upper()
        self.log_file = os.getenv("AUTOPERF_LOG_FILE")
        
        # Mode debug
        self.debug_mode = os.getenv("AUTOPERF_DEBUG", "false").lower() == "true"
    
    def get_kernel_file_path(self, kernel_type: str) -> Path:
        """Retourne le chemin vers le fichier du kernel"""
        if kernel_type not in self.kernel_types:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        basename = self.kernel_basenames[kernel_type]
        return self.cpp_root / "src" / basename
    
    def get_default_prompt_paths(self) -> List[Path]:
        """Retourne la liste des chemins vers les prompts par défaut"""
        default_prompts = ["algorithmic.txt", "parallelization.txt", "vectorization.txt"]
        return [self.prompts_dir / prompt_name for prompt_name in default_prompts]
    
    def validate(self) -> List[str]:
        """Valide la configuration et retourne une liste d'erreurs"""
        errors = []
        
        # Vérifier que les répertoires essentiels existent
        if not self.repo_root.exists():
            errors.append(f"Repository root not found: {self.repo_root}")
        
        if not self.cpp_root.exists():
            errors.append(f"C++ source directory not found: {self.cpp_root}")
        
        # Vérifier les kernels
        for kernel_type, basename in self.kernel_basenames.items():
            kernel_path = self.cpp_root / "src" / basename
            if not kernel_path.exists():
                errors.append(f"Kernel file not found: {kernel_path}")
        
        # Vérifier les valeurs numériques
        if self.default_temperature < 0 or self.default_temperature > 2:
            errors.append(f"Invalid temperature: {self.default_temperature}")
        
        if self.default_phases < 1:
            errors.append(f"Invalid phases: {self.default_phases}")
        
        if self.default_branching < 1:
            errors.append(f"Invalid branching: {self.default_branching}")
        
        if self.default_jobs < 1:
            errors.append(f"Invalid jobs: {self.default_jobs}")
        
        if self.build_type not in ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]:
            errors.append(f"Invalid build type: {self.build_type}")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append(f"Invalid log level: {self.log_level}")
        
        return errors
    
    def to_dict(self) -> dict:
        """Convertit la configuration en dictionnaire"""
        return {
            "paths": {
                "repo_root": str(self.repo_root),
                "cpp_root": str(self.cpp_root),
                "build_dir": str(self.build_dir),
                "runs_dir": str(self.runs_dir),
                "best_dir": str(self.best_dir),
                "prompts_dir": str(self.prompts_dir),
                "env_path": str(self.env_path),
                "orch_src_dir": str(self.orch_src_dir),
            },
            "kernels": {
                "types": self.kernel_types,
                "basenames": self.kernel_basenames,
            },
            "llm": {
                "default_model": self.default_model,
                "default_temperature": self.default_temperature,
            },
            "optimization": {
                "default_phases": self.default_phases,
                "default_branching": self.default_branching,
                "default_jobs": self.default_jobs,
            },
            "build": {
                "enable_benchmarks": self.enable_benchmarks,
                "enable_tests": self.enable_tests,
                "build_type": self.build_type,
            },
            "logging": {
                "log_level": self.log_level,
                "log_file": self.log_file,
            },
            "debug_mode": self.debug_mode,
        }


# Instance globale
_config_instance: Optional[SimpleConfig] = None


def get_config() -> SimpleConfig:
    """Retourne l'instance globale de configuration"""
    global _config_instance
    if _config_instance is None:
        _config_instance = SimpleConfig()
    return _config_instance


def reload_config() -> SimpleConfig:
    """Recharge la configuration"""
    global _config_instance
    _config_instance = SimpleConfig()
    return _config_instance


# Fonction de compatibilité avec l'ancienne API
def get_legacy_config() -> dict:
    """Retourne la configuration dans le format legacy"""
    config = get_config()
    return {
        'REPO_ROOT': str(config.repo_root),
        'CPP_ROOT': str(config.cpp_root),
        'BUILD_DIR': str(config.build_dir),
        'RUNS_DIR': str(config.runs_dir),
        'BEST_DIR': str(config.best_dir),
        'PROMPTS_DIR': str(config.prompts_dir),
        'ENV_PATH': str(config.env_path),
        'ORCH_SRC_DIR': str(config.orch_src_dir),
        'KERNEL_TYPES': config.kernel_types,
        'KERNEL_BASENAMES': config.kernel_basenames,
    }


# Alias pour compatibilité
get_settings = get_config