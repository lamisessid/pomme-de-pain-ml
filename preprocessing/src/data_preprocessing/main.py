"""
Point d'entrée principal du preprocessing.
Orchestre le chargement, nettoyage et transformation des données.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from .config import data_config, processing_config, logging_config
from .utils import setup_logger, print_summary, save_file
from .loaders.sales_loader import SalesLoader
from .loaders.weather_loader import WeatherLoader
from .loaders.events_loader import EventsLoader
from .loaders.products_loader import ProductsLoader


class DataPreprocessingPipeline:
    """Pipeline complet de preprocessing."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger("DataPipeline")
        self.sales_data = None
        self.weather_data = None
        self.events_data = None
        self.products_data = None
        self.final_dataset = None
        
        self.logger.info("🚀 Initialisation du pipeline de preprocessing")
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """
        Exécute le pipeline complet.
        
        Étapes:
        1. Chargement des données
        2. Nettoyage
        3. Feature engineering
        4. Fusion
        5. Split train/test
        """
        self.logger.info("🎬 Démarrage du pipeline complet")
        
        try:
            # Étape 1: Chargement
            self.load_all_data()
            
            # Étape 2: Nettoyage
            self.clean_all_data()
            
            # Étape 3: Feature engineering
            self.create_features()
            
            # Étape 4: Fusion
            self.merge_data()
            
            # Étape 5: Split
            self.split_data()
            
            self.logger.info("✅ Pipeline exécuté avec succès")
            
            return self.final_dataset
        
        except Exception as e:
            self.logger.error(f"❌ Erreur dans le pipeline: {e}", exc_info=True)
            raise
    
    def load_all_data(self) -> None:
        """Charge toutes les sources de données."""
        self.logger.info("📂 Étape 1: Chargement des données")
        
        # Ventes
        self.logger.info("  📊 Chargement des ventes...")
        sales_loader = SalesLoader(self.logger)
        self.sales_data = sales_loader.load()
        print_summary(self.sales_data, "Ventes")
        
        # Météo
        self.logger.info("  🌤️  Chargement de la météo...")
        weather_loader = WeatherLoader(self.logger)
        self.weather_data = weather_loader.load()
        print_summary(self.weather_data, "Météo")
        
        # Événements
        self.logger.info("  🎭 Chargement des événements...")
        events_loader = EventsLoader(self.logger)
        self.events_data = events_loader.load()
        print_summary(self.events_data, "Événements")
        
        # Produits
        self.logger.info("  🛒 Chargement des produits...")
        products_loader = ProductsLoader(self.logger)
        self.products_data = products_loader.load()
        print_summary(self.products_data, "Produits")
        
        self.logger.info("✅ Données chargées")
    
    def clean_all_data(self) -> None:
        """Nettoie toutes les données."""
        self.logger.info("🧹 Étape 2: Nettoyage des données")
        
        # Importer les cleaners ici pour éviter les imports circulaires
        from .cleaners.sales_cleaner import SalesCleaner
        from .cleaners.weather_cleaner import WeatherCleaner
        from .cleaners.events_cleaner import EventsCleaner
        from .cleaners.products_cleaner import ProductsCleaner
        
        # Nettoyer les ventes
        if self.sales_data is not None:
            self.logger.info("  📊 Nettoyage des ventes...")
            sales_cleaner = SalesCleaner(self.logger)
            self.sales_data = sales_cleaner.clean(self.sales_data)
            sales_cleaner.save(self.sales_data, data_config.sales_cleaned)
        
        # Nettoyer la météo
        if self.weather_data is not None:
            self.logger.info("  🌤️  Nettoyage de la météo...")
            weather_cleaner = WeatherCleaner(self.logger)
            self.weather_data = weather_cleaner.clean(self.weather_data)
            weather_cleaner.save(self.weather_data, data_config.weather_cleaned)
        
        # Nettoyer les événements
        if self.events_data is not None:
            self.logger.info("  🎭 Nettoyage des événements...")
            events_cleaner = EventsCleaner(self.logger)
            self.events_data = events_cleaner.clean(self.events_data)
            events_cleaner.save(self.events_data, data_config.events_cleaned)
        
        # Nettoyer les produits
        if self.products_data is not None:
            self.logger.info("  🛒 Nettoyage des produits...")
            products_cleaner = ProductsCleaner(self.logger)
            self.products_data = products_cleaner.clean(self.products_data)
            products_cleaner.save(self.products_data, data_config.products_cleaned)
        
        self.logger.info("✅ Données nettoyées")
    
    def create_features(self) -> None:
        """Crée les features pour chaque source."""
        self.logger.info("🔧 Étape 3: Feature engineering")
        
        from .transformers.temporal_transformer import TemporalTransformer
        from .transformers.weather_transformer import WeatherTransformer
        from .transformers.events_transformer import EventsTransformer
        
        # Features temporelles pour les ventes
        if self.sales_data is not None and 'date' in self.sales_data.columns:
            self.logger.info("  📅 Création des features temporelles...")
            temporal_transformer = TemporalTransformer(self.logger)
            self.sales_data = temporal_transformer.transform(self.sales_data)
        
        # Features météo
        if self.weather_data is not None and 'date' in self.weather_data.columns:
            self.logger.info("  🌡️  Création des features météo...")
            weather_transformer = WeatherTransformer(self.logger)
            self.weather_data = weather_transformer.transform(self.weather_data)
        
        # Features événements
        if self.events_data is not None and 'date' in self.events_data.columns:
            self.logger.info("  🎪 Création des features événements...")
            events_transformer = EventsTransformer(self.logger)
            self.events_data = events_transformer.transform(self.events_data)
        
        self.logger.info("✅ Features créées")
    
    def merge_data(self) -> None:
        """Fusionne toutes les sources en un dataset unique."""
        self.logger.info("🔗 Étape 4: Fusion des données")
        
        from .merger import DataMerger
        
        merger = DataMerger(self.logger)
        self.final_dataset = merger.merge(
            sales_data=self.sales_data,
            weather_data=self.weather_data,
            events_data=self.events_data,
            products_data=self.products_data
        )
        
        if self.final_dataset is not None:
            print_summary(self.final_dataset, "Dataset final")
            save_file(self.final_dataset, data_config.final_dataset, self.logger)
            self.logger.info("✅ Données fusionnées")
    
    def split_data(self) -> None:
        """Split les données en train/validation/test."""
        self.logger.info("✂️  Étape 5: Split des données")
        
        from .splitter import DataSplitter
        
        if self.final_dataset is None:
            self.logger.warning("Aucune donnée à splitter")
            return
        
        splitter = DataSplitter(self.logger)
        train_data, val_data, test_data = splitter.split(self.final_dataset)
        
        # Sauvegarder les splits
        if train_data is not None:
            save_file(train_data, data_config.train_dataset, self.logger)
        if val_data is not None:
            save_file(val_data, data_config.validation_dataset, self.logger)
        if test_data is not None:
            save_file(test_data, data_config.test_dataset, self.logger)
        
        self.logger.info("✅ Données splittées")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Retourne des informations sur l'état du pipeline."""
        return {
            'sales_loaded': self.sales_data is not None,
            'weather_loaded': self.weather_data is not None,
            'events_loaded': self.events_data is not None,
            'products_loaded': self.products_data is not None,
            'final_dataset_created': self.final_dataset is not None,
            'final_shape': self.final_dataset.shape if self.final_dataset is not None else None
        }


def main():
    """Fonction principale exécutable."""
    logger = setup_logger("Main")
    logger.info("🎬 Démarrage du preprocessing Pomme de Pain")
    
    try:
        # Initialiser et exécuter le pipeline
        pipeline = DataPreprocessingPipeline(logger)
        final_data = pipeline.run_full_pipeline()
        
        logger.info("✨ Preprocessing terminé avec succès!")
        logger.info(f"📦 Dataset final shape: {final_data.shape}")
        logger.info(f"💾 Sauvegardé dans: {data_config.final_dataset}")
        
        # Afficher un résumé
        print_summary(final_data, "DATASET FINAL")
        
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()