"""
Module pour splitter les données en train/validation/test.
"""

from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np
import logging

from .config import processing_config
from .utils import setup_logger


class DataSplitter:
    """Split les données selon différentes stratégies."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger(self.__class__.__name__)
        self.split_stats = {}
    
    def split(self, data: pd.DataFrame, method: str = 'temporal') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split les données en train, validation et test.
        
        Méthodes disponibles:
        - 'temporal': Split chronologique (recommandé pour séries temporelles)
        - 'random': Split aléatoire
        - 'stratified': Split stratifié par catégorie ou saison
        """
        self.logger.info("✂️  Split des données...")
        
        if data is None or data.empty:
            self.logger.error("❌ Données vides pour le split")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        if 'date' not in data.columns and method == 'temporal':
            self.logger.warning("⚠️  Colonne 'date' manquante, utilisation du split aléatoire")
            method = 'random'
        
        if method == 'temporal':
            return self._temporal_split(data)
        elif method == 'random':
            return self._random_split(data)
        elif method == 'stratified':
            return self._stratified_split(data)
        else:
            self.logger.warning(f"⚠️  Méthode '{method}' non reconnue, utilisation du split temporel")
            return self._temporal_split(data)
    
    def _temporal_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split chronologique (pour séries temporelles)."""
        self.logger.info("   📅 Split temporel...")
        
        # S'assurer que les données sont triées par date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Convertir les dates de configuration
        train_end = pd.to_datetime(processing_config.train_end)
        validation_end = pd.to_datetime(processing_config.validation_end)
        
        # Dates min et max des données
        min_date = data['date'].min()
        max_date = data['date'].max()
        
        self.logger.info(f"     - Période totale: {min_date.date()} to {max_date.date()}")
        self.logger.info(f"     - Fin train: {train_end.date()}")
        self.logger.info(f"     - Fin validation: {validation_end.date()}")
        
        # Split
        train_data = data[data['date'] <= train_end].copy()
        val_data = data[(data['date'] > train_end) & (data['date'] <= validation_end)].copy()
        test_data = data[data['date'] > validation_end].copy()
        
        # Vérifier que les splits ne sont pas vides
        if train_data.empty:
            self.logger.error("❌ Dataset d'entraînement vide")
            raise ValueError("Dataset d'entraînement vide")
        
        # Statistiques
        self._calculate_split_stats(train_data, val_data, test_data)
        
        return train_data, val_data, test_data
    
    def _random_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split aléatoire."""
        self.logger.info("   🎲 Split aléatoire...")
        
        # Proportions par défaut: 70% train, 15% val, 15% test
        train_ratio = 0.7
        val_ratio = 0.15
        # test_ratio = 0.15 (le reste)
        
        # Mélanger les données
        data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculer les indices de split
        n_total = len(data_shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split
        train_data = data_shuffled.iloc[:n_train].copy()
        val_data = data_shuffled.iloc[n_train:n_train + n_val].copy()
        test_data = data_shuffled.iloc[n_train + n_val:].copy()
        
        # Statistiques
        self._calculate_split_stats(train_data, val_data, test_data)
        
        return train_data, val_data, test_data
    
    def _stratified_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split stratifié (par saison ou catégorie)."""
        self.logger.info("   🎯 Split stratifié...")
        
        # Déterminer la colonne de stratification
        stratify_by = None
        
        if 'season' in data.columns:
            stratify_by = 'season'
            self.logger.info("     - Stratification par saison")
        elif 'category' in data.columns:
            stratify_by = 'category'
            self.logger.info("     - Stratification par catégorie produit")
        elif 'month' in data.columns:
            stratify_by = 'month'
            self.logger.info("     - Stratification par mois")
        else:
            self.logger.warning("⚠️  Aucune colonne de stratification trouvée, utilisation du split aléatoire")
            return self._random_split(data)
        
        # Proportions
        train_ratio = 0.7
        val_ratio = 0.15
        
        # Initialiser les DataFrames vides
        train_data = pd.DataFrame()
        val_data = pd.DataFrame()
        test_data = pd.DataFrame()
        
        # Split par groupe
        for group_name, group_data in data.groupby(stratify_by):
            n_group = len(group_data)
            n_train = int(n_group * train_ratio)
            n_val = int(n_group * val_ratio)
            
            # Mélanger le groupe
            group_shuffled = group_data.sample(frac=1, random_state=42)
            
            # Ajouter aux splits
            train_data = pd.concat([train_data, group_shuffled.iloc[:n_train]])
            val_data = pd.concat([val_data, group_shuffled.iloc[n_train:n_train + n_val]])
            test_data = pd.concat([test_data, group_shuffled.iloc[n_train + n_val:]])
        
        # Mélanger à l'intérieur de chaque split
        train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
        test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Vérifier la distribution
        self._check_stratification(data, train_data, val_data, test_data, stratify_by)
        
        # Statistiques
        self._calculate_split_stats(train_data, val_data, test_data)
        
        return train_data, val_data, test_data
    
    def _calculate_split_stats(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        """Calcule les statistiques des splits."""
        n_train = len(train_data)
        n_val = len(val_data)
        n_test = len(test_data)
        n_total = n_train + n_val + n_test
        
        if n_total == 0:
            return
        
        train_pct = n_train / n_total * 100
        val_pct = n_val / n_total * 100
        test_pct = n_test / n_total * 100
        
        self.split_stats.update({
            'train_samples': n_train,
            'val_samples': n_val,
            'test_samples': n_test,
            'train_percentage': train_pct,
            'val_percentage': val_pct,
            'test_percentage': test_pct,
            'total_samples': n_total
        })
        
        self.logger.info(f"     - Train: {n_train} échantillons ({train_pct:.1f}%)")
        self.logger.info(f"     - Validation: {n_val} échantillons ({val_pct:.1f}%)")
        self.logger.info(f"     - Test: {n_test} échantillons ({test_pct:.1f}%)")
        
        # Périodes temporelles si disponibles
        for name, df in [('Train', train_data), ('Validation', val_data), ('Test', test_data)]:
            if 'date' in df.columns and not df.empty:
                min_date = df['date'].min()
                max_date = df['date'].max()
                self.logger.info(f"     - {name}: {min_date.date()} to {max_date.date()}")
    
    def _check_stratification(self, full_data: pd.DataFrame, train_data: pd.DataFrame, 
                             val_data: pd.DataFrame, test_data: pd.DataFrame, stratify_col: str) -> None:
        """Vérifie que la stratification est préservée."""
        self.logger.info(f"   🔍 Vérification stratification par {stratify_col}...")
        
        # Distributions
        full_dist = full_data[stratify_col].value_counts(normalize=True).sort_index()
        train_dist = train_data[stratify_col].value_counts(normalize=True).sort_index()
        val_dist = val_data[stratify_col].value_counts(normalize=True).sort_index()
        test_dist = test_data[stratify_col].value_counts(normalize=True).sort_index()
        
        # Calculer les écarts
        train_diff = (train_dist - full_dist).abs().mean()
        val_diff = (val_dist - full_dist).abs().mean()
        test_diff = (test_dist - full_dist).abs().mean()
        
        self.logger.info(f"     - Écart moyen train: {train_diff:.3f}")
        self.logger.info(f"     - Écart moyen val: {val_diff:.3f}")
        self.logger.info(f"     - Écart moyen test: {test_diff:.3f}")
        
        if train_diff > 0.1 or val_diff > 0.1 or test_diff > 0.1:
            self.logger.warning("⚠️  Écarts importants dans la stratification")
        
        self.split_stats['stratification_train_diff'] = train_diff
        self.split_stats['stratification_val_diff'] = val_diff
        self.split_stats['stratification_test_diff'] = test_diff
    
    def save_splits(self, train_data: pd.DataFrame, val_data: pd.DataFrame, 
                   test_data: pd.DataFrame, output_dir: str) -> None:
        """Sauvegarde les splits dans des fichiers séparés."""
        from .utils import save_file
        
        self.logger.info("💾 Sauvegarde des splits...")
        
        # Chemins de sauvegarde
        train_path = output_dir / "train.parquet"
        val_path = output_dir / "validation.parquet"
        test_path = output_dir / "test.parquet"
        
        # Sauvegarde
        if not train_data.empty:
            save_file(train_data, train_path, self.logger)
        
        if not val_data.empty:
            save_file(val_data, val_path, self.logger)
        
        if not test_data.empty:
            save_file(test_data, test_path, self.logger)
    
    def get_split_info(self) -> Dict:
        """Retourne les informations sur le split."""
        return self.split_stats