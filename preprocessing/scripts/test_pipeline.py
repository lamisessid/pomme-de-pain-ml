# scripts/test_pipeline.py
#!/usr/bin/env python3
"""
Script de test pour le pipeline de preprocessing.
"""

import sys
from pathlib import Path

# Ajouter le src au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preprocessing.main import DataPreprocessingPipeline
from src.data_preprocessing.utils import setup_logger


def test_pipeline():
    """Teste le pipeline complet."""
    logger = setup_logger("TestPipeline")
    
    try:
        logger.info("🧪 Démarrage du test du pipeline...")
        
        # Initialiser le pipeline
        pipeline = DataPreprocessingPipeline(logger)
        
        # Tester chaque étape
        logger.info("\n1. Test du chargement des données...")
        pipeline.load_all_data()
        
        logger.info("\n2. Test du nettoyage...")
        pipeline.clean_all_data()
        
        logger.info("\n3. Test de la création des features...")
        pipeline.create_features()
        
        logger.info("\n4. Test de la fusion...")
        pipeline.merge_data()
        
        logger.info("\n5. Test du split...")
        pipeline.split_data()
        
        # Afficher les informations
        info = pipeline.get_pipeline_info()
        logger.info("\n📊 Informations du pipeline:")
        for key, value in info.items():
            logger.info(f"   - {key}: {value}")
        
        logger.info("✅ Test du pipeline réussi!")
        
    except Exception as e:
        logger.error(f"❌ Erreur pendant le test: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    test_pipeline()