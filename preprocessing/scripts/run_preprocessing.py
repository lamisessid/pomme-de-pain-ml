#!/usr/bin/env python3
"""
Script pour exécuter le pipeline de preprocessing.
Usage: python scripts/run_preprocessing.py
"""

import sys
from pathlib import Path

# Ajouter le src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing.main import main

if __name__ == "__main__":
    print("🚀 Lancement du preprocessing Pomme de Pain...")
    main()
    print("✅ Preprocessing terminé!")