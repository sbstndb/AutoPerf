#!/usr/bin/env python3
"""
Script de test pour valider le système de thinking.
"""
import sys
import os

# Ajouter le répertoire racine du projet au PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from orchestrator.src.thinking_config import ThinkingConfig, ThinkingMode


def test_thinking_config():
    """Test de la classe ThinkingConfig"""
    print("=== Test ThinkingConfig ===")
    
    # Test disabled mode
    config_disabled = ThinkingConfig(mode=ThinkingMode.DISABLED)
    print(f"Disabled: {config_disabled}")
    print(f"Instructions: {config_disabled.get_thinking_instructions()}")
    print()
    
    # Test dynamic mode
    config_dynamic = ThinkingConfig(mode=ThinkingMode.DYNAMIC)
    print(f"Dynamic: {config_dynamic}")
    print(f"Instructions: {config_dynamic.get_thinking_instructions()[:100]}...")
    print()
    
    # Test budget mode
    config_budget = ThinkingConfig(mode=ThinkingMode.BUDGET, budget=300)
    print(f"Budget: {config_budget}")
    print(f"Instructions: {config_budget.get_thinking_instructions()[:100]}...")
    print()


def test_cli_args_simulation():
    """Simule les arguments CLI pour tester from_cli_args"""
    print("=== Test CLI Args Simulation ===")
    
    class MockArgs:
        def __init__(self, **kwargs):
            self.thinking_disabled = kwargs.get('thinking_disabled', False)
            self.thinking_dynamic = kwargs.get('thinking_dynamic', False)
            self.thinking_budget = kwargs.get('thinking_budget', None)
    
    # Test différents arguments CLI
    test_cases = [
        {"thinking_disabled": True},
        {"thinking_dynamic": True},
        {"thinking_budget": 500},
        {},  # Cas par défaut
    ]
    
    for i, case in enumerate(test_cases):
        args = MockArgs(**case)
        config = ThinkingConfig.from_cli_args(args)
        print(f"Test case {i+1}: {case} -> {config}")
    print()


def test_prompt_context():
    """Test de la génération de contexte pour les prompts"""
    print("=== Test Prompt Context ===")
    
    configs = [
        ThinkingConfig(mode=ThinkingMode.DISABLED),
        ThinkingConfig(mode=ThinkingMode.DYNAMIC),
        ThinkingConfig(mode=ThinkingMode.BUDGET, budget=400)
    ]
    
    for config in configs:
        context = config.to_prompt_context()
        print(f"{config} -> Context: {context}")
    print()


def test_thinking_extraction():
    """Test de l'extraction du code avec thinking tags"""
    print("=== Test Thinking Extraction ===")
    
    from orchestrator.src.utils import extract_code_from_markdown
    
    test_texts = [
        # Cas normal sans thinking
        "```cpp\nint main() { return 0; }\n```",
        
        # Cas avec thinking tags
        """<thinking>
        I need to optimize this loop by unrolling it.
        The current approach has too many branch predictions.
        </thinking>
        
        ```cpp
        int optimized_function() {
            // Optimized code here
            return 42;
        }
        ```""",
        
        # Cas avec thinking tags mais sans markdown
        """<thinking>
        Analysis of the problem...
        </thinking>
        
        int direct_code() {
            return 1;
        }"""
    ]
    
    for i, text in enumerate(test_texts):
        extracted = extract_code_from_markdown(text)
        print(f"Test {i+1}:")
        print(f"Input: {text[:50]}...")
        print(f"Extracted: {extracted[:50]}...")
        print()


if __name__ == "__main__":
    test_thinking_config()
    test_cli_args_simulation() 
    test_prompt_context()
    test_thinking_extraction()
    print("✅ Tous les tests sont terminés!")