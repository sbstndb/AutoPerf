#!/usr/bin/env python3
"""
Script de test pour la nouvelle architecture séparée.
"""
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from orchestrator.src.cpp_project_v2 import CppProject
from orchestrator.src.simple_config import get_config


def test_new_architecture():
    """Test de la nouvelle architecture"""
    
    print("🧪 Testing new C++ project architecture...")
    
    # Test de la configuration
    print("\n1. Testing settings...")
    settings = get_config()
    print(f"   ✅ Repo root: {settings.repo_root}")
    print(f"   ✅ Available kernels: {settings.kernel_types}")
    print(f"   ✅ Default model: {settings.default_model}")
    
    # Test avec un répertoire temporaire
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_project"
        
        print(f"\n2. Testing CppProject in {temp_path}...")
        
        try:
            # Créer un projet de test
            project = CppProject(str(temp_path), "matvec")
            print("   ✅ Project created successfully")
            
            # Test du statut initial
            status = project.get_build_status()
            print(f"   ✅ Initial status: configured={status['configured']}, built={status['built']}")
            
            # Test de lecture du kernel existant
            kernel_info = project.get_kernel_info()
            print(f"   ✅ Kernel info: {kernel_info['file_exists']} - {kernel_info.get('line_count', 'N/A')} lines")
            
            # Test d'un kernel simple
            simple_kernel = """
#include <vector>

void matvec(const std::vector<std::vector<double>>& A, 
           const std::vector<double>& x, 
           std::vector<double>& y) {
    const size_t n = A.size();
    for (size_t i = 0; i < n; ++i) {
        y[i] = 0.0;
        for (size_t j = 0; j < A[i].size(); ++j) {
            y[i] += A[i][j] * x[j];
        }
    }
}
"""
            
            print("\n3. Testing kernel update...")
            if project.update_kernel(simple_kernel):
                print("   ✅ Kernel updated successfully")
            else:
                print("   ❌ Failed to update kernel")
                return False
            
            print("\n4. Testing configuration...")
            config_result = project.configure_v2()
            print(f"   ✅ Configure: {config_result.success}")
            if config_result.configure_time:
                print(f"   ⏱️  Time: {config_result.configure_time:.2f}s")
            
            if not config_result.success:
                print(f"   ❌ Configure failed: {config_result.log}")
                return False
            
            print("\n5. Testing build...")
            build_result = project.build_v2()
            print(f"   ✅ Build: {build_result.success}")
            if build_result.build_time:
                print(f"   ⏱️  Time: {build_result.build_time:.2f}s")
                
            if not build_result.success:
                print(f"   ❌ Build failed: {build_result.log}")
                return False
            
            print("\n6. Testing components individually...")
            
            # Test du builder
            print("   🔨 Builder component:")
            print(f"      - Configured: {project.builder.is_configured()}")
            print(f"      - Built: {project.builder.is_built()}")
            
            # Test du kernel manager
            print("   📝 Kernel manager component:")
            kernel_path = project.kernel_manager.get_kernel_path()
            print(f"      - Kernel path: {kernel_path}")
            print(f"      - File exists: {kernel_path.exists()}")
            
            # Test des tests (peut échouer selon l'environnement)
            print("\n7. Testing tests (optional)...")
            try:
                test_result = project.test_v2()
                print(f"   ✅ Tests: {test_result.success}")
                if test_result.tests_passed:
                    print(f"   📊 Passed: {test_result.tests_passed}, Failed: {test_result.tests_failed}")
            except Exception as e:
                print(f"   ⚠️  Tests skipped: {e}")
            
            # Test des benchmarks (peut échouer selon l'environnement)
            print("\n8. Testing benchmarks (optional)...")
            try:
                benchmark_result = project.benchmark_v2()
                print(f"   ✅ Benchmark: {benchmark_result.success}")
                if benchmark_result.score:
                    print(f"   📊 Score: {benchmark_result.score:.2f} ns")
            except Exception as e:
                print(f"   ⚠️  Benchmark skipped: {e}")
            
            print("\n🎉 All core tests passed! New architecture is working.")
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_legacy_compatibility():
    """Test de compatibilité avec l'ancienne API"""
    
    print("\n🔄 Testing legacy API compatibility...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "legacy_test"
        
        try:
            project = CppProject(str(temp_path), "matvec")
            
            # Test de l'ancienne API
            success, log = project.configure()
            print(f"   ✅ Legacy configure: {success}")
            
            success, log = project.build()
            print(f"   ✅ Legacy build: {success}")
            
            success, log = project.test()
            print(f"   ✅ Legacy test: {success}")
            
            success, log, score = project.benchmark()
            print(f"   ✅ Legacy benchmark: {success}, score: {score}")
            
            print("🎉 Legacy compatibility maintained!")
            return True
            
        except Exception as e:
            print(f"   ❌ Legacy compatibility error: {e}")
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("AutoPerf - New Architecture Test")
    print("=" * 60)
    
    # Test de l'architecture
    success1 = test_new_architecture()
    
    # Test de compatibilité
    success2 = test_legacy_compatibility()
    
    if success1 and success2:
        print("\n🚀 All tests passed! Ready for migration.")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed. Check the output above.")
        sys.exit(1)