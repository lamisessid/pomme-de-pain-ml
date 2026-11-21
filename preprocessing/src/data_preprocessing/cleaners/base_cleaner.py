"""
Classe de base pour tous les cleaners.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import logging

from ..config import processing_config
from ..utils import setup_logger, save_file, detect_outliers_iqr


class BaseCleaner(ABC):
    """Classe abstraite pour le nettoyage des données."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger(self.__class__.__name__)
        self.stats = {}
    
    @abstractmethod
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les données."""
        pass
    
    def save(self, data: pd.DataFrame, file_path: str) -> None:
        """Sauvegarde les données nettoyées."""
        save_file(data, file_path, self.logger)
        self.logger.info(f"✅ Données nettoyées sauvegardées: {file_path}")
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de nettoyage."""
        return self.stats
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
        """
        Gère les valeurs manquantes.
        
        Stratégies:
        - 'forward_fill': propagation vers l'avant
        - 'backward_fill': propagation vers l'arrière
        - 'mean': moyenne
        - 'median': médiane
        - 'interpolate': interpolation linéaire
        - 'drop': supprimer les lignes
        """
        data = data.copy()
        
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                before = data[col].isnull().sum()
                
                if strategy == 'forward_fill':
                    data[col] = data[col].ffill()
                elif strategy == 'backward_fill':
                    data[col] = data[col].bfill()
                elif strategy == 'mean':
                    data[col] = data[col].fillna(data[col].mean())
                elif strategy == 'median':
                    data[col] = data[col].fillna(data[col].median())
                elif strategy == 'interpolate':
                    data[col] = data[col].interpolate()
                elif strategy == 'drop':
                    data = data.dropna(subset=[col])
                else:
                    self.logger.warning(f"Stratégie {strategy} non supportée pour {col}")
                
                after = data[col].isnull().sum()
                self.logger.info(f"   - {col}: {before} → {after} valeurs manquantes")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, columns: list, method: str = 'clip') -> pd.DataFrame:
        """
        Gère les outliers.
        
        Méthodes:
        - 'clip': limiter aux bornes
        - 'remove': supprimer les lignes
        - 'median': remplacer par la médiane
        """
        data = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Détecter les outliers avec IQR
            outlier_info = detect_outliers_iqr(data, col)
            
            if outlier_info['outlier_count'] > 0:
                self.logger.info(f"   - {col}: {outlier_info['outlier_count']} outliers détectés")
                
                if method == 'clip':
                    # Limiter les valeurs aux bornes
                    data[col] = data[col].clip(
                        lower=outlier_info['lower_bound'],
                        upper=outlier_info['upper_bound']
                    )
                    self.logger.info(f"     → Valeurs limitées aux bornes")
                
                elif method == 'remove':
                    # Supprimer les lignes avec outliers
                    mask = (data[col] >= outlier_info['lower_bound']) & (data[col] <= outlier_info['upper_bound'])
                    data = data[mask]
                    self.logger.info(f"     → {outlier_info['outlier_count']} lignes supprimées")
                
                elif method == 'median':
                    # Remplacer par la médiane
                    median_val = data[col].median()
                    outlier_mask = (data[col] < outlier_info['lower_bound']) | (data[col] > outlier_info['upper_bound'])
                    data.loc[outlier_mask, col] = median_val
                    self.logger.info(f"     → Outliers remplacés par médiane: {median_val}")
                
                # Enregistrer les statistiques
                if col not in self.stats:
                    self.stats[col] = {}
                self.stats[col]['outliers'] = outlier_info
        
        return data