"""
Configuration et gestion du système de thinking pour l'optimisation.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any


class ThinkingMode(Enum):
    """Modes de thinking disponibles"""
    DISABLED = "disabled"
    DYNAMIC = "dynamic"
    BUDGET = "budget"


@dataclass
class ThinkingConfig:
    """Configuration du système de thinking"""
    mode: ThinkingMode
    budget: Optional[int] = None
    
    @classmethod
    def from_cli_args(cls, args) -> 'ThinkingConfig':
        """Crée une configuration à partir des arguments CLI"""
        if args.thinking_disabled:
            return cls(mode=ThinkingMode.DISABLED)
        elif args.thinking_dynamic:
            return cls(mode=ThinkingMode.DYNAMIC)
        elif args.thinking_budget is not None:
            return cls(mode=ThinkingMode.BUDGET, budget=args.thinking_budget)
        else:
            # Par défaut : disabled
            return cls(mode=ThinkingMode.DISABLED)
    
    @classmethod
    def from_settings(cls, settings) -> 'ThinkingConfig':
        """Crée une configuration à partir des settings"""
        mode_str = settings.llm.thinking_mode.lower()
        
        if mode_str == "disabled":
            return cls(mode=ThinkingMode.DISABLED)
        elif mode_str == "dynamic":
            return cls(mode=ThinkingMode.DYNAMIC)
        elif mode_str == "budget":
            return cls(mode=ThinkingMode.BUDGET, budget=settings.llm.thinking_budget)
        else:
            raise ValueError(f"Invalid thinking mode: {mode_str}")
    
    def to_prompt_context(self) -> Dict[str, Any]:
        """Convertit la configuration en contexte pour les prompts"""
        context = {
            "thinking_enabled": self.mode != ThinkingMode.DISABLED,
            "thinking_mode": self.mode.value
        }
        
        if self.mode == ThinkingMode.BUDGET and self.budget:
            context["thinking_budget"] = self.budget
            
        return context
    
    def get_thinking_instructions(self) -> str:
        """Retourne les instructions de thinking selon le mode"""
        if self.mode == ThinkingMode.DISABLED:
            return ""
        
        base_instruction = """
Before providing your solution, think through the problem step by step within <thinking> tags.
Your thinking should include:
- Analysis of the current code's performance bottlenecks
- Consideration of different optimization strategies
- Evaluation of trade-offs between approaches
- Step-by-step reasoning for your chosen solution

After your thinking, provide only the optimized C++ code without the thinking tags.
"""
        
        if self.mode == ThinkingMode.DYNAMIC:
            return base_instruction + """
Use as much thinking space as needed to thoroughly analyze the problem.
"""
        
        elif self.mode == ThinkingMode.BUDGET:
            return base_instruction + f"""
Limit your thinking to approximately {self.budget} tokens.
Be concise but thorough in your analysis.
"""
        
        return base_instruction
    
    def __str__(self) -> str:
        """Représentation string de la configuration"""
        if self.mode == ThinkingMode.BUDGET:
            return f"ThinkingConfig(mode={self.mode.value}, budget={self.budget})"
        return f"ThinkingConfig(mode={self.mode.value})"


def validate_thinking_budget(budget: Optional[int]) -> bool:
    """Valide un budget de thinking"""
    if budget is None:
        return True
    return 0 <= budget <= 2000


def estimate_thinking_tokens(text: str) -> int:
    """Estime approximativement le nombre de tokens dans un texte de thinking"""
    # Approximation simple : ~4 caractères par token
    return len(text) // 4