"""
Composant pour l'exécution sécurisée de commandes système.
"""
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple
from orchestrator.src.build_interfaces import ICommandRunner
from orchestrator.src.errors import BuildError


class CommandRunner:
    """Exécuteur de commandes système avec gestion d'erreurs"""
    
    def __init__(self, default_cwd: Optional[Path] = None):
        self.default_cwd = default_cwd
    
    def run_command(self, command: list[str], cwd: Optional[Path] = None) -> Tuple[bool, str]:
        """
        Exécute une commande et retourne (success, log)
        
        Args:
            command: Liste des arguments de la commande
            cwd: Répertoire de travail (optionnel)
            
        Returns:
            Tuple (success, log_output)
            
        Raises:
            BuildError: En cas d'erreur OS
        """
        work_dir = cwd or self.default_cwd
        
        try:
            start_time = time.time()
            result = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            execution_time = time.time() - start_time
            
            success = result.returncode == 0
            log = result.stdout + result.stderr
            
            # Ajouter des informations de debug si nécessaire
            if execution_time > 1.0:  # Log si l'exécution prend plus d'une seconde
                log += f"\n[DEBUG] Command executed in {execution_time:.2f}s"
            
            return success, log
            
        except OSError as e:
            raise BuildError(f"OS error executing command {' '.join(command)}: {e}") from e
        except Exception as e:
            raise BuildError(f"Unexpected error executing command {' '.join(command)}: {e}") from e


class TimedCommandRunner(CommandRunner):
    """Version avec mesure de temps automatique"""
    
    def run_command(self, command: list[str], cwd: Optional[Path] = None) -> Tuple[bool, str, float]:
        """
        Exécute une commande et retourne (success, log, execution_time)
        """
        work_dir = cwd or self.default_cwd
        
        try:
            start_time = time.time()
            result = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            execution_time = time.time() - start_time
            
            success = result.returncode == 0
            log = result.stdout + result.stderr
            log += f"\n[TIMING] Executed in {execution_time:.2f}s"
            
            return success, log, execution_time
            
        except OSError as e:
            raise BuildError(f"OS error executing command {' '.join(command)}: {e}") from e
        except Exception as e:
            raise BuildError(f"Unexpected error executing command {' '.join(command)}: {e}") from e