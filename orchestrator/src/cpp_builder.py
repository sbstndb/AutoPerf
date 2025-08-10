"""
Composant responsable de la configuration et compilation des projets C++.
"""
import shutil
from pathlib import Path
from typing import Optional
from orchestrator.src.build_interfaces import ICppBuilder, ConfigureResult, BuildResult
from orchestrator.src.command_runner import TimedCommandRunner
from orchestrator.src.errors import ConfigureError, BuildError


class CppBuilder:
    """Gestionnaire de build pour les projets C++ avec CMake"""
    
    def __init__(
        self, 
        source_dir: Path, 
        build_dir: Path, 
        kernel_type: str,
        command_runner: Optional[TimedCommandRunner] = None
    ):
        self.source_dir = Path(source_dir)
        self.build_dir = Path(build_dir)
        self.kernel_type = kernel_type
        self.command_runner = command_runner or TimedCommandRunner(default_cwd=self.source_dir.parent)
        
        # Créer le répertoire de build
        self.build_dir.mkdir(parents=True, exist_ok=True)
    
    def configure(self, enable_benchmarks: bool = True, enable_tests: bool = True) -> ConfigureResult:
        """
        Configure le projet CMake pour le kernel spécifié
        
        Args:
            enable_benchmarks: Activer la compilation des benchmarks
            enable_tests: Activer la compilation des tests
            
        Returns:
            ConfigureResult avec les détails de la configuration
            
        Raises:
            ConfigureError: En cas d'échec de la configuration
        """
        # Nettoyer le répertoire de build pour éviter les conflits de cache CMake
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        # Construire les arguments CMake
        cmake_args = [
            f"-S{self.source_dir}",
            f"-B{self.build_dir}",
            f"-DAUTOPERF_KERNELS={self.kernel_type}",
            f"-DAUTOPERF_ENABLE_BENCHES={'ON' if enable_benchmarks else 'OFF'}",
            f"-DAUTOPERF_ENABLE_TESTS={'ON' if enable_tests else 'OFF'}",
            "-DCMAKE_BUILD_TYPE=Release",  # Toujours en Release pour les benchmarks
        ]
        
        command = ["cmake"] + cmake_args
        
        try:
            success, log, configure_time = self.command_runner.run_command(command)
            
            if not success:
                raise ConfigureError(f"CMake configuration failed:\n{log}")
            
            return ConfigureResult(
                success=True,
                log=log,
                configure_time=configure_time
            )
            
        except Exception as e:
            if isinstance(e, ConfigureError):
                raise
            raise ConfigureError(f"Configuration error: {e}") from e
    
    def build(self, target: Optional[str] = None) -> BuildResult:
        """
        Build le projet ou une target spécifique
        
        Args:
            target: Target spécifique à builder (optionnel)
            
        Returns:
            BuildResult avec les détails du build
            
        Raises:
            BuildError: En cas d'échec du build
        """
        if not self.build_dir.exists():
            raise BuildError("Build directory does not exist. Run configure() first.")
        
        # Construire la commande de build
        command = ["cmake", "--build", str(self.build_dir)]
        if target:
            command.extend(["--target", target])
        
        # Ajouter le parallélisme selon la configuration
        from orchestrator.src.settings import get_settings
        settings = get_settings()
        command.extend(["--parallel", str(settings.build.parallel_jobs)])
        
        try:
            success, log, build_time = self.command_runner.run_command(command)
            
            if not success:
                raise BuildError(f"Build failed:\n{log}")
            
            return BuildResult(
                success=True,
                log=log,
                build_time=build_time
            )
            
        except Exception as e:
            if isinstance(e, BuildError):
                raise
            raise BuildError(f"Build error: {e}") from e
    
    def clean(self) -> bool:
        """
        Nettoie les fichiers de build
        
        Returns:
            True si le nettoyage a réussi
        """
        try:
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)
            return True
        except Exception:
            return False
    
    def build_target(self, target: str) -> BuildResult:
        """
        Build une target spécifique (alias pour build avec target)
        
        Args:
            target: Nom de la target à builder
            
        Returns:
            BuildResult
        """
        return self.build(target=target)
    
    def get_executable_path(self, executable_name: str) -> Path:
        """
        Retourne le chemin vers un exécutable buildé
        
        Args:
            executable_name: Nom de l'exécutable
            
        Returns:
            Path vers l'exécutable
        """
        return self.build_dir / "bin" / executable_name
    
    def is_configured(self) -> bool:
        """
        Vérifie si le projet est configuré
        
        Returns:
            True si CMakeCache.txt existe
        """
        return (self.build_dir / "CMakeCache.txt").exists()
    
    def is_built(self) -> bool:
        """
        Vérifie si le projet a été buildé
        
        Returns:
            True si le répertoire bin existe et contient des fichiers
        """
        bin_dir = self.build_dir / "bin"
        return bin_dir.exists() and any(bin_dir.iterdir())


class OptimizedCppBuilder(CppBuilder):
    """Version optimisée avec cache et build incrémental"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_configure_hash = None
        self._last_build_hash = None
    
    def configure_if_needed(self, enable_benchmarks: bool = True, enable_tests: bool = True) -> ConfigureResult:
        """
        Configure seulement si nécessaire (détection de changements)
        """
        config_hash = hash((enable_benchmarks, enable_tests, self.kernel_type))
        
        if self.is_configured() and config_hash == self._last_configure_hash:
            return ConfigureResult(
                success=True,
                log="Configuration skipped (no changes detected)",
                configure_time=0.0
            )
        
        result = self.configure(enable_benchmarks, enable_tests)
        if result.success:
            self._last_configure_hash = config_hash
        
        return result
    
    def build_if_needed(self, target: Optional[str] = None) -> BuildResult:
        """
        Build seulement si nécessaire
        """
        build_hash = hash((target, self.kernel_type))
        
        if self.is_built() and build_hash == self._last_build_hash:
            return BuildResult(
                success=True,
                log="Build skipped (no changes detected)",
                build_time=0.0
            )
        
        result = self.build(target)
        if result.success:
            self._last_build_hash = build_hash
        
        return result