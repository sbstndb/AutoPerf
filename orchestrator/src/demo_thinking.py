#!/usr/bin/env python3
"""
Démonstration du système de thinking avec exemples concrets.
"""
import sys
import os

# Ajouter le répertoire racine du projet au PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from orchestrator.src.thinking_config import ThinkingConfig, ThinkingMode


def demo_thinking_instructions():
    """Démontre les différentes instructions de thinking"""
    print("=" * 60)
    print("DÉMONSTRATION DU SYSTÈME DE THINKING")
    print("=" * 60)
    
    configs = [
        ("Mode Désactivé", ThinkingConfig(mode=ThinkingMode.DISABLED)),
        ("Mode Dynamique", ThinkingConfig(mode=ThinkingMode.DYNAMIC)),
        ("Mode Budget (500 tokens)", ThinkingConfig(mode=ThinkingMode.BUDGET, budget=500)),
    ]
    
    for name, config in configs:
        print(f"\n🔹 {name}")
        print("-" * 40)
        instructions = config.get_thinking_instructions()
        if instructions.strip():
            print(instructions.strip())
        else:
            print("Aucune instruction de thinking (mode désactivé)")
        print()


def demo_cli_usage():
    """Démontre l'utilisation en ligne de commande"""
    print("=" * 60)
    print("EXEMPLES D'UTILISATION CLI")
    print("=" * 60)
    
    examples = [
        ("Optimisation basique (sans thinking)", 
         "python -m orchestrator.src.main --kernel matmul"),
        
        ("Thinking dynamique avec prompts spécialisés",
         "python -m orchestrator.src.main --kernel matmul --thinking-dynamic \\"),
         
        ("  --prompts orchestrator/prompts/algorithmic_thinking.txt,orchestrator/prompts/vectorization_thinking.txt",),
        
        ("Thinking avec budget fixe",
         "python -m orchestrator.src.main --kernel matvec --thinking-budget 400 \\"),
         
        ("  --jobs 4 --phases 3 --branching 2",),
        
        ("Comparaison performance (thinking vs baseline)",
         "# Avec thinking:"),
         
        ("python -m orchestrator.src.main --kernel reduce --thinking-budget 300",),
        
        ("# Sans thinking:",),
        
        ("python -m orchestrator.src.main --kernel reduce --thinking-disabled",),
    ]
    
    for i, example in enumerate(examples, 1):
        if len(example) == 2:
            title, command = example
            print(f"\n🔹 {title}")
            print("-" * 40)
            print(f"{command}")
        else:
            print(f"{example[0]}")


def demo_expected_output():
    """Montre des exemples de sortie avec thinking"""
    print("=" * 60)
    print("EXEMPLES DE SORTIE AVEC THINKING")
    print("=" * 60)
    
    print("\n🔹 Exemple de réponse avec thinking dynamique")
    print("-" * 40)
    
    example_with_thinking = """<thinking>
Je dois analyser ce kernel de multiplication matricielle pour identifier les bottlenecks.

Analyse actuelle :
1. La boucle externe itère sur les lignes de A
2. La boucle du milieu itère sur les colonnes de B  
3. La boucle interne fait le produit scalaire

Problèmes identifiés :
- Accès mémoire non optimal : B[k][j] provoque des cache misses
- Pas de blocking pour optimiser la cache
- Pas de vectorisation explicite

Solutions possibles :
1. Réorganiser les boucles (ijk -> ikj) pour améliorer la localité
2. Ajouter du loop tiling/blocking
3. Utiliser la vectorisation avec AVX

Je vais implémenter la réorganisation des boucles comme première optimisation.
</thinking>

void kernel_matmul(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}"""
    
    print(example_with_thinking)
    
    print("\n🔹 Code extrait (après nettoyage des balises thinking)")
    print("-" * 40)
    
    from orchestrator.src.utils import extract_code_from_markdown
    cleaned = extract_code_from_markdown(example_with_thinking)
    print(cleaned)


def demo_performance_tips():
    """Conseils pour optimiser l'utilisation du thinking"""
    print("=" * 60)
    print("CONSEILS D'OPTIMISATION")
    print("=" * 60)
    
    tips = [
        ("💡 Choisir le bon mode", [
            "- Mode désactivé : pour des tests rapides ou des kernels simples",
            "- Mode dynamique : pour des optimisations complexes avec GPT-4",
            "- Mode budget : compromis qualité/coût pour usage en production"
        ]),
        
        ("💡 Budgets recommandés", [
            "- 200-300 tokens : optimisations algorithmiques simples",
            "- 400-600 tokens : usage général (recommandé)",  
            "- 700-1000 tokens : problèmes très complexes"
        ]),
        
        ("💡 Prompts spécialisés", [
            "- algorithmic_thinking.txt : pour les changements algorithmiques",
            "- vectorization_thinking.txt : pour SIMD/AVX optimizations",
            "- parallelization_thinking.txt : pour OpenMP/threading"
        ]),
        
        ("💡 Monitoring", [
            "- Comparez les résultats avec/sans thinking",
            "- Surveillez le coût en tokens vs amélioration performance",
            "- Ajustez le budget selon les résultats obtenus"
        ])
    ]
    
    for title, items in tips:
        print(f"\n{title}")
        print("-" * 40)
        for item in items:
            print(f"  {item}")


if __name__ == "__main__":
    demo_thinking_instructions()
    demo_cli_usage() 
    demo_expected_output()
    demo_performance_tips()
    
    print("\n" + "=" * 60)
    print("✅ DÉMONSTRATION TERMINÉE")
    print("=" * 60)
    print("\nPour commencer, essayez :")
    print("python -m orchestrator.src.main --kernel matmul --thinking-budget 400 --help")