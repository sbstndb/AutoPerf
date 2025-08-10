"""
Composant responsable de la gestion des kernels (code source).
"""
from pathlib import Path
from typing import Optional
from orchestrator.src.build_interfaces import IKernelManager
from orchestrator.src.simple_config import get_config


class KernelManager:
    """Gestionnaire pour les fichiers de kernels C++"""
    
    def __init__(self, source_dir: Path, kernel_type: str):
        self.source_dir = Path(source_dir)
        self.kernel_type = kernel_type
        self.settings = get_config()
        
        # Valider le type de kernel
        if kernel_type not in self.settings.kernel_types:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def update_kernel(self, kernel_code: str) -> bool:
        """
        Met à jour le code du kernel
        
        Args:
            kernel_code: Nouveau code C++ du kernel
            
        Returns:
            True si la mise à jour a réussi
        """
        kernel_file = self.get_kernel_path()
        
        try:
            # Créer le répertoire parent si nécessaire
            kernel_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Écrire le nouveau code
            with open(kernel_file, 'w', encoding='utf-8') as f:
                f.write(kernel_code)
            
            return True
            
        except Exception as e:
            print(f"Failed to update kernel: {e}")
            return False
    
    def get_kernel_path(self) -> Path:
        """
        Retourne le chemin vers le fichier kernel
        
        Returns:
            Path vers le fichier kernel
        """
        basename = self.settings.kernel_basenames[self.kernel_type]
        return self.source_dir / "src" / basename
    
    def read_kernel_code(self) -> Optional[str]:
        """
        Lit le code actuel du kernel
        
        Returns:
            Code du kernel ou None si le fichier n'existe pas
        """
        kernel_file = self.get_kernel_path()
        
        try:
            if kernel_file.exists():
                with open(kernel_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Failed to read kernel: {e}")
        
        return None
    
    def backup_kernel(self, backup_suffix: str = ".backup") -> bool:
        """
        Crée une sauvegarde du kernel actuel
        
        Args:
            backup_suffix: Suffixe pour le fichier de sauvegarde
            
        Returns:
            True si la sauvegarde a réussi
        """
        kernel_file = self.get_kernel_path()
        backup_file = kernel_file.with_suffix(kernel_file.suffix + backup_suffix)
        
        try:
            if kernel_file.exists():
                import shutil
                shutil.copy2(kernel_file, backup_file)
                return True
        except Exception as e:
            print(f"Failed to backup kernel: {e}")
        
        return False
    
    def restore_kernel(self, backup_suffix: str = ".backup") -> bool:
        """
        Restaure le kernel depuis une sauvegarde
        
        Args:
            backup_suffix: Suffixe du fichier de sauvegarde
            
        Returns:
            True si la restauration a réussi
        """
        kernel_file = self.get_kernel_path()
        backup_file = kernel_file.with_suffix(kernel_file.suffix + backup_suffix)
        
        try:
            if backup_file.exists():
                import shutil
                shutil.copy2(backup_file, kernel_file)
                return True
        except Exception as e:
            print(f"Failed to restore kernel: {e}")
        
        return False
    
    def validate_kernel_code(self, kernel_code: str) -> tuple[bool, list[str]]:
        """
        Valide le code du kernel (vérifications de base)
        
        Args:
            kernel_code: Code à valider
            
        Returns:
            Tuple (is_valid, list_of_issues)
        """
        issues = []
        
        # Vérifications de base
        if not kernel_code.strip():
            issues.append("Kernel code is empty")
        
        # Vérifier la présence d'includes dangereux
        dangerous_includes = ['<cstdlib>', '<system>', '<exec']
        for include in dangerous_includes:
            if include in kernel_code:
                issues.append(f"Potentially dangerous include: {include}")
        
        # Vérifier la présence de fonctions dangereuses
        dangerous_functions = ['system(', 'exec(', 'popen(']
        for func in dangerous_functions:
            if func in kernel_code:
                issues.append(f"Potentially dangerous function call: {func}")
        
        # Vérifier la syntaxe de base C++
        if not self._basic_cpp_syntax_check(kernel_code):
            issues.append("Basic C++ syntax issues detected")
        
        return len(issues) == 0, issues
    
    def _basic_cpp_syntax_check(self, code: str) -> bool:
        """
        Vérification basique de la syntaxe C++
        
        Args:
            code: Code à vérifier
            
        Returns:
            True si la syntaxe semble correcte
        """
        # Vérifications très basiques
        brace_count = code.count('{') - code.count('}')
        paren_count = code.count('(') - code.count(')')
        
        return brace_count == 0 and paren_count == 0
    
    def get_kernel_info(self) -> dict:
        """
        Retourne des informations sur le kernel
        
        Returns:
            Dictionnaire avec les informations
        """
        kernel_file = self.get_kernel_path()
        
        info = {
            "kernel_type": self.kernel_type,
            "file_path": str(kernel_file),
            "file_exists": kernel_file.exists(),
            "source_dir": str(self.source_dir),
        }
        
        if kernel_file.exists():
            try:
                stat = kernel_file.stat()
                info.update({
                    "file_size": stat.st_size,
                    "last_modified": stat.st_mtime,
                })
                
                # Compter les lignes de code
                with open(kernel_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    info["line_count"] = len(lines)
                    info["non_empty_lines"] = len([l for l in lines if l.strip()])
                    
            except Exception as e:
                info["read_error"] = str(e)
        
        return info


class SafeKernelManager(KernelManager):
    """Version sécurisée avec validation renforcée"""
    
    def update_kernel(self, kernel_code: str) -> bool:
        """
        Met à jour le kernel après validation
        
        Args:
            kernel_code: Nouveau code du kernel
            
        Returns:
            True si la mise à jour a réussi
        """
        # Valider avant de sauvegarder
        is_valid, issues = self.validate_kernel_code(kernel_code)
        
        if not is_valid:
            print(f"Kernel validation failed: {issues}")
            return False
        
        # Créer une sauvegarde avant la mise à jour
        self.backup_kernel()
        
        # Effectuer la mise à jour
        success = super().update_kernel(kernel_code)
        
        if not success:
            # Restaurer la sauvegarde en cas d'échec
            self.restore_kernel()
        
        return success