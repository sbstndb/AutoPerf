#!/usr/bin/env python3
"""
Script d'exécution pour AutoPerf qui résout les problèmes d'import.
"""
import sys
import os
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    from orchestrator.src.main import main
    main()