import os
import re

# Removed unused run_cmd; CppProject._run_command is the single runner


def load_env_file(env_path):
    """Loads environment variables from a .env file."""
    if not os.path.exists(env_path):
        return
    with open(env_path, 'r', encoding='utf-8') as f:
        for raw in f.read().splitlines():
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"')
            if key and value and key not in os.environ:
                os.environ[key] = value


def _strip_reasoning_tags(text: str) -> str:
    cleaned = text
    tags = [
        "think",
        "thinking",
        "thought",
        "thoughts",
        "reasoning",
        "chain_of_thought",
        "cot",
    ]
    for tag in tags:
        cleaned = re.sub(fr"<\s*{tag}[^>]*?>[\s\S]*?<\s*/\s*{tag}\s*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(fr"<\s*/?\s*{tag}[^>]*?>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|[^|]*?\|>", "", cleaned)
    return cleaned


def extract_code_from_markdown(text):
    """Extracts the first C++ code block from a markdown string and strips reasoning tags."""
    # D'abord supprimer les balises de thinking
    text = _strip_reasoning_tags(text)
    
    # Si on a des blocs de code markdown, extraire le premier
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            code_block = parts[1]
            lines = code_block.splitlines()
            # Enlever la première ligne si c'est un indicateur de langage
            if lines and lines[0].strip().lower() in ("cpp", "c++", "c", ""):
                lines = lines[1:]
            return "\n".join(lines).strip()
    
    # Sinon, supposer que tout le texte restant est du code
    return text.strip()

