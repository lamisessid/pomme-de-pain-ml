"""
Cleaner pour les données de ventes.
"""

from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import logging

from .base_cleaner import BaseCleaner
from ..config import processing_config


class SalesCleaner(BaseCleaner):
    """Nettoie les données de ventes."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.stats = {
            'rows_removed': 0,
            'rows_added': 0,
            'outliers_handled': 0,
            'missing_filled': 0
        }
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données de ventes.
        
        Étapes:
        1. Valider la structure
        2. Gérer les dates manquantes
        3. Gérer les valeurs manquantes
        4. Gérer les outliers
        5. Gérer les quantités négatives
        6. Créer des features dérivées
        """
        self.logger.info("🧹 Nettoyage des données de ventes...")
        
        if data is None or data.empty:
            self.logger.warning("⚠️  Données de ventes vides")
            return data
        
        data = data.copy()
        initial_rows = len(data)
        
        # 1. Valider la structure de base
        data = self._validate_structure(data)
        
        # 2. S'assurer que 'date' est datetime et trier
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data = data.sort_values('date')
            
            # Supprimer les dates invalides
            invalid_dates = data['date'].isnull().sum()
            if invalid_dates > 0:
                self.logger.warning(f"Suppression de {invalid_dates} lignes avec dates invalides")
                data = data.dropna(subset=['date'])
        
        # 3. Gérer les valeurs manquantes
        data = self._handle_missing_sales(data)
        
        # 4. Gérer les outliers pour les quantités
        if 'quantity' in data.columns:
            data = self._handle_outliers(
                data, 
                columns=['quantity'],
                method='clip'  # Limiter plutôt que supprimer pour garder l'historique
            )
        
        # 5. Gérer les quantités négatives
        data = self._handle_negative_quantities(data)
        
        # 6. Créer des indicateurs de données manquantes
        data = self._create_missing_indicators(data)
        
        # 7. Calculer des statistiques par produit
        data = self._calculate_product_stats(data)
        
        # 8. Filtrer sur la période configurée
        data = self._filter_by_date_range(data)
        
        # Statistiques finales
        final_rows = len(data)
        self.stats['rows_removed'] = initial_rows - final_rows
        self.stats['final_rows'] = final_rows
        
        self.logger.info(f"✅ Nettoyage terminé: {initial_rows} → {final_rows} lignes")
        self._log_cleaning_stats()
        
        return data
    
    def _validate_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Valide et corrige la structure de base."""
        # S'assurer que les colonnes critiques existent
        critical_cols = ['date', 'product_id', 'quantity']
        
        for col in critical_cols:
            if col not in data.columns:
                self.logger.error(f"❌ Colonne critique manquante: {col}")
                raise ValueError(f"Colonne {col} manquante")
        
        # Supprimer les lignes où les colonnes critiques sont manquantes
        before = len(data)
        data = data.dropna(subset=critical_cols)
        after = len(data)
        
        if before != after:
            self.logger.info(f"   - {before - after} lignes supprimées (valeurs critiques manquantes)")
        
        # S'assurer que product_id est string
        data['product_id'] = data['product_id'].astype(str)
        
        # S'assurer que quantity est numérique
        data['quantity'] = pd.to_numeric(data['quantity'], errors='coerce')
        
        return data
    
    def _handle_missing_sales(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Gère les ventes manquantes.
        
        Stratégie:
        - Pour chaque produit, créer une série temporelle complète
        - Remplir les jours manquants avec 0 ou NaN selon contexte
        """
        if 'date' not in data.columns or 'product_id' not in data.columns:
            return data
        
        self.logger.info("   📅 Traitement des jours manquants par produit...")
        
        # Identifier tous les produits uniques
        unique_products = data['product_id'].unique()
        self.logger.info(f"     - {len(unique_products)} produits uniques")
        
        # Identifier la plage de dates complète (basée sur les dates valides)
        valid_dates = data['date'].dropna()
        if valid_dates.empty:
            self.logger.warning("⚠️  Aucune date valide trouvée")
            return data
        
        min_date = valid_dates.min()
        max_date = valid_dates.max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        
        self.logger.info(f"     - Période: {min_date.date()} to {max_date.date()} ({len(all_dates)} jours)")
        
        # SOLUTION CORRIGÉE : Créer un DataFrame avec toutes les combinaisons date × produit
        # Sans utiliser MultiIndex problématique
        date_product_combinations = []
        for date in all_dates:
            for product_id in unique_products:
                date_product_combinations.append({
                    'date': date,
                    'product_id': product_id
                })
        
        all_combinations_df = pd.DataFrame(date_product_combinations)
        
        # Fusionner avec les données existantes
        data_complete = pd.merge(
            all_combinations_df,
            data,
            on=['date', 'product_id'],
            how='left'
        )
        
        # Remplir les quantités manquantes avec 0 (pas de vente ce jour-là)
        if 'quantity' in data_complete.columns:
            data_complete['quantity'] = data_complete['quantity'].fillna(0)
            filled_count = data_complete['quantity'].isnull().sum()
            if filled_count == 0:
                self.logger.info(f"     - Toutes les quantités complétées (0 pour jours sans vente)")
            else:
                self.logger.warning(f"     - {filled_count} quantités toujours manquantes après remplissage")
        
        # Remplir les autres colonnes numériques avec forward/backward fill par produit
        numeric_cols = data_complete.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['date', 'product_id', 'quantity']]
        
        for col in numeric_cols:
            if data_complete[col].isnull().any():
                # Méthode robuste pour remplir les valeurs manquantes par produit
                def fill_missing_values(group):
                    # Forward fill puis backward fill
                    filled = group.ffill().bfill()
                    return filled
                
                # Appliquer par groupe de produit
                data_complete[col] = data_complete.groupby('product_id')[col].transform(fill_missing_values)
                
                # Compter combien ont été remplis
                filled = data_complete[col].notna().sum() - data[col].notna().sum() if col in data.columns else 0
                if filled > 0:
                    self.logger.info(f"     - {col}: {filled} valeurs manquantes remplies par produit")
        
        # Remplir les colonnes catégorielles avec la valeur la plus fréquente par produit
        categorical_cols = data_complete.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['date', 'product_id']]
        
        for col in categorical_cols:
            if data_complete[col].isnull().any() and col in data.columns:
                # Remplacer par la valeur la plus fréquente (mode) par produit
                def fill_with_mode(group):
                    if group.isnull().all():
                        return group
                    mode_values = group.mode()
                    if not mode_values.empty:
                        # Remplir avec la valeur la plus fréquente
                        return group.fillna(mode_values.iloc[0])
                    return group
                
                data_complete[col] = data_complete.groupby('product_id')[col].transform(fill_with_mode)
        
        # Vérifier la structure finale
        expected_rows = len(all_dates) * len(unique_products)
        actual_rows = len(data_complete)
        
        if expected_rows == actual_rows:
            self.logger.info(f"     ✅ Structure complète: {actual_rows} lignes (jours × produits)")
        else:
            self.logger.warning(f"     ⚠️  Structure incomplète: {actual_rows} lignes au lieu de {expected_rows}")
        
        # Mettre à jour les statistiques
        rows_added = actual_rows - len(data) if len(data) > 0 else actual_rows
        self.stats['rows_added'] = rows_added
        self.stats['missing_filled'] = data_complete.isnull().sum().sum()
        
        return data_complete
    
    def _handle_negative_quantities(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gère les quantités de ventes négatives."""
        if 'quantity' not in data.columns:
            return data
        
        negative_count = (data['quantity'] < 0).sum()
        
        if negative_count > 0:
            self.logger.warning(f"   ⚠️  {negative_count} quantités négatives détectées")
            
            # Pour l'instant, on met à 0 (supposant que c'est une erreur)
            data.loc[data['quantity'] < 0, 'quantity'] = 0
            self.logger.info(f"     → Mises à 0")
            
            self.stats['negative_quantities'] = negative_count
        
        return data
    
    def _create_missing_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des indicateurs pour les données manquantes/imputées."""
        # Indicateur pour les ventes imputées (0 quand on a rempli les manquants)
        if 'quantity' in data.columns:
            # On pourrait garder une trace, mais pour l'instant simple
            pass
        
        # Indicateur pour les autres colonnes manquantes
        for col in data.columns:
            if col not in ['date', 'product_id'] and data[col].isnull().any():
                indicator_name = f"{col}_was_missing"
                data[indicator_name] = data[col].isnull().astype(int)
        
        return data
    
    def _calculate_product_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule des statistiques par produit."""
        if 'quantity' not in data.columns:
            return data
        
        # Statistiques de base par produit
        product_stats = data.groupby('product_id').agg({
            'quantity': ['sum', 'mean', 'std', 'min', 'max']
        }).round(2)
        
        # Aplatir les colonnes
        product_stats.columns = [f'product_{col[0]}_{col[1]}' for col in product_stats.columns]
        product_stats = product_stats.reset_index()
        
        # Fusionner avec les données originales
        data = pd.merge(data, product_stats, on='product_id', how='left')
        
        return data
    
    def _filter_by_date_range(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filtre les données sur la période configurée."""
        if 'date' not in data.columns:
            return data
        
        # Convertir les dates de configuration
        start_date = pd.to_datetime(processing_config.start_date)
        end_date = pd.to_datetime(processing_config.end_date)
        
        # Filtrer
        before = len(data)
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        after = len(data)
        
        if before != after:
            self.logger.info(f"   - Filtrage période: {before - after} lignes enlevées")
            self.logger.info(f"     → Gardé {start_date.date()} to {end_date.date()}")
        
        return data
    
    def _log_cleaning_stats(self) -> None:
        """Affiche les statistiques de nettoyage."""
        self.logger.info("📊 Statistiques de nettoyage des ventes:")
        for key, value in self.stats.items():
            if isinstance(value, (int, float)) and value > 0:
                self.logger.info(f"   - {key}: {value}")