"""
Composant responsable de l'exécution des tests C++.
"""
import re
from pathlib import Path
from typing import Optional
from orchestrator.src.build_interfaces import ICppTester, TestResult
from orchestrator.src.command_runner import TimedCommandRunner
from orchestrator.src.errors import TestError


class CppTester:
    """Gestionnaire des tests pour les projets C++ avec CTest"""
    
    def __init__(
        self,
        build_dir: Path,
        kernel_type: str,
        command_runner: Optional[TimedCommandRunner] = None
    ):
        self.build_dir = Path(build_dir)
        self.kernel_type = kernel_type
        self.command_runner = command_runner or TimedCommandRunner(default_cwd=self.build_dir.parent)
    
    def run_tests(self, test_pattern: Optional[str] = None) -> TestResult:
        """
        Exécute les tests avec CTest
        
        Args:
            test_pattern: Pattern pour filtrer les tests (optionnel)
            
        Returns:
            TestResult avec les détails de l'exécution
            
        Raises:
            TestError: En cas d'échec des tests
        """
        if not self.build_dir.exists():
            raise TestError("Build directory does not exist. Build the project first.")
        
        # Construire la commande CTest
        command = ["ctest", "--test-dir", str(self.build_dir), "--output-on-failure"]
        
        # Ajouter le pattern si spécifié
        if test_pattern:
            command.extend(["-R", test_pattern])
        
        # Ajouter la verbosité pour plus de détails
        command.append("--verbose")
        
        try:
            success, log, test_time = self.command_runner.run_command(command)
            
            # Parser les résultats des tests
            tests_passed, tests_failed = self._parse_test_results(log)
            
            if not success:
                raise TestError(f"Tests failed:\n{log}")
            
            return TestResult(
                success=True,
                log=log,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                test_time=test_time
            )
            
        except Exception as e:
            if isinstance(e, TestError):
                raise
            raise TestError(f"Test execution error: {e}") from e
    
    def run_specific_test(self, test_name: str) -> TestResult:
        """
        Exécute un test spécifique
        
        Args:
            test_name: Nom exact du test à exécuter
            
        Returns:
            TestResult
        """
        return self.run_tests(test_pattern=f"^{re.escape(test_name)}$")
    
    def run_kernel_tests(self) -> TestResult:
        """
        Exécute uniquement les tests du kernel actuel
        
        Returns:
            TestResult
        """
        # Les tests suivent généralement le pattern test_<kernel> ou tests_<kernel>
        test_patterns = [
            f"test_{self.kernel_type}",
            f"tests_{self.kernel_type}",
            f"Test{self.kernel_type.capitalize()}",
        ]
        
        # Essayer chaque pattern jusqu'à trouver des tests
        for pattern in test_patterns:
            try:
                result = self.run_tests(test_pattern=pattern)
                if result.tests_passed and result.tests_passed > 0:
                    return result
            except TestError:
                continue
        
        # Si aucun pattern spécifique ne fonctionne, exécuter tous les tests
        return self.run_tests()
    
    def list_available_tests(self) -> list[str]:
        """
        Liste tous les tests disponibles
        
        Returns:
            Liste des noms de tests
        """
        if not self.build_dir.exists():
            return []
        
        command = ["ctest", "--test-dir", str(self.build_dir), "--show-only=json-v1"]
        
        try:
            success, log, _ = self.command_runner.run_command(command)
            if success:
                return self._parse_test_list(log)
        except Exception:
            pass
        
        return []
    
    def _parse_test_results(self, log: str) -> tuple[Optional[int], Optional[int]]:
        """
        Parse les résultats des tests depuis la sortie CTest
        
        Args:
            log: Sortie de CTest
            
        Returns:
            Tuple (tests_passed, tests_failed)
        """
        tests_passed = None
        tests_failed = None
        
        # Patterns pour parser les résultats CTest
        patterns = [
            r"(\d+)/(\d+) Test\s+#\d+:",  # Format standard CTest
            r"Total Tests:\s*(\d+)",       # Total des tests
            r"Passed:\s*(\d+)",            # Tests réussis
            r"Failed:\s*(\d+)",            # Tests échoués
        ]
        
        # Chercher le pattern de résumé final
        summary_pattern = r"(\d+)% tests passed, (\d+) tests failed out of (\d+)"
        summary_match = re.search(summary_pattern, log)
        
        if summary_match:
            total_tests = int(summary_match.group(3))
            tests_failed = int(summary_match.group(2))
            tests_passed = total_tests - tests_failed
        else:
            # Fallback: compter les lignes de tests individuels
            test_lines = re.findall(r"^\s*\d+/\d+ Test\s+#\d+:\s+\w+\s+\.\.\.\s+(\w+)", log, re.MULTILINE)
            if test_lines:
                tests_passed = sum(1 for result in test_lines if result == "Passed")
                tests_failed = sum(1 for result in test_lines if result == "Failed")
        
        return tests_passed, tests_failed
    
    def _parse_test_list(self, json_output: str) -> list[str]:
        """
        Parse la liste des tests depuis la sortie JSON de CTest
        
        Args:
            json_output: Sortie JSON de CTest
            
        Returns:
            Liste des noms de tests
        """
        try:
            import json
            data = json.loads(json_output)
            tests = data.get("tests", [])
            return [test.get("name", "") for test in tests if test.get("name")]
        except (json.JSONDecodeError, KeyError):
            # Fallback: parser la sortie texte
            test_names = re.findall(r"Test\s+#\d+:\s+(\w+)", json_output)
            return test_names
    
    def get_test_executable_path(self, test_name: str) -> Optional[Path]:
        """
        Retourne le chemin vers l'exécutable d'un test
        
        Args:
            test_name: Nom du test
            
        Returns:
            Path vers l'exécutable ou None si introuvable
        """
        possible_paths = [
            self.build_dir / "bin" / test_name,
            self.build_dir / "bin" / f"test_{test_name}",
            self.build_dir / "tests" / test_name,
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                return path
        
        return None


class DetailedCppTester(CppTester):
    """Version étendue avec analyses détaillées des tests"""
    
    def run_tests_with_coverage(self, test_pattern: Optional[str] = None) -> TestResult:
        """
        Exécute les tests avec mesure de couverture de code (si disponible)
        
        Args:
            test_pattern: Pattern pour filtrer les tests
            
        Returns:
            TestResult avec informations de couverture
        """
        # TODO: Intégrer gcov/lcov pour la couverture de code
        result = self.run_tests(test_pattern)
        
        # Placeholder pour la couverture de code
        # Cette fonctionnalité pourrait être ajoutée plus tard
        
        return result
    
    def run_tests_with_valgrind(self, test_pattern: Optional[str] = None) -> TestResult:
        """
        Exécute les tests avec Valgrind pour détecter les fuites mémoire
        
        Args:
            test_pattern: Pattern pour filtrer les tests
            
        Returns:
            TestResult avec analyse Valgrind
        """
        # Vérifier si Valgrind est disponible
        try:
            success, _, _ = self.command_runner.run_command(["which", "valgrind"])
            if not success:
                return self.run_tests(test_pattern)  # Fallback sans Valgrind
        except Exception:
            return self.run_tests(test_pattern)  # Fallback sans Valgrind
        
        # Construire la commande avec Valgrind
        command = [
            "ctest", "--test-dir", str(self.build_dir), 
            "--output-on-failure",
            "--overwrite", "MemoryCheckCommand=valgrind",
            "--overwrite", "MemoryCheckCommandOptions=--leak-check=full --show-leak-kinds=all",
            "-T", "memcheck"
        ]
        
        if test_pattern:
            command.extend(["-R", test_pattern])
        
        try:
            success, log, test_time = self.command_runner.run_command(command)
            
            tests_passed, tests_failed = self._parse_test_results(log)
            
            return TestResult(
                success=success,
                log=log,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                test_time=test_time
            )
            
        except Exception as e:
            raise TestError(f"Valgrind test execution error: {e}") from e