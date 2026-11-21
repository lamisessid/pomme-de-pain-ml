"""
Loader spécifique pour les données de ventes.
"""

from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import logging

from .base_loader import BaseLoader
from ..config import data_config
from ..utils import load_file


class SalesLoader(BaseLoader):
    """Charge et valide les données de ventes."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.expected_columns = data_config.expected_sales_columns
        self.primary_key = ['date', 'product_id']
    
    def load(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Charge les données de ventes.
        
        Stratégie:
        1. Essayer le fichier Parquet (plus rapide)
        2. Sinon essayer le CSV
        3. Valider la structure
        """
        self.logger.info("Chargement des données de ventes...")
        
        # Déterminer le fichier source
        if file_path is None:
            # Essayer Parquet d'abord
            if data_config.sales_parquet.exists():
                file_path = data_config.sales_parquet
                self.logger.info("Utilisation du fichier Parquet")
            elif data_config.sales_raw.exists():
                file_path = data_config.sales_raw
                self.logger.info("Utilisation du fichier CSV")
            else:
                self.logger.error("Aucun fichier de ventes trouvé")
                raise FileNotFoundError("Fichier de ventes introuvable")
        
        # Charger les données
        self.data = load_file(file_path, self.logger)
        
        # Nettoyer les noms de colonnes
        self._clean_column_names()
        
        # Détecter et convertir les dates
        date_columns = self._detect_date_columns()
        if date_columns:
            self._convert_dates(date_columns)
        else:
            self.logger.warning("Aucune colonne de date détectée")
        
        # Valider
        if self.validate():
            self.logger.info("✅ Données de ventes chargées avec succès")
            self._update_metadata()
        else:
            self.logger.warning("⚠️  Données de ventes chargées avec des problèmes")
        
        return self.data
    
    def _clean_column_names(self) -> None:
        """Nettoie les noms de colonnes."""
        if self.data is None:
            return
        
        # Standardisation des noms
        column_mapping = {
    # Vos colonnes réelles
    'produit_id': 'product_id',
    'ventes_unites': 'quantity',
    'pertes_unites': 'losses',
    'chiffre_affaires': 'revenue',  
    'cout_pertes': 'cost_losses',
    'temperature': 'temperature',
    'pluie_mm': 'precipitation',  
    'ensoleillement': 'sunshine', 
    'jour_semaine': 'day_of_week',
    'mois': 'month',
    'est_weekend': 'is_weekend',
    'est_ferie': 'is_holiday',
    'a_evenement': 'has_event',
    'types_evenements': 'event_types',
    'impact_evenement': 'event_impact',
    
    # Noms standards
    'Date': 'date',
    'date': 'date',
}
        
        # Appliquer le mapping
        self.data.columns = [
            column_mapping.get(col.strip(), col.strip().lower())
            for col in self.data.columns
        ]
        
        self.logger.info(f"Colonnes après nettoyage: {list(self.data.columns)}")
    
    def validate(self) -> bool:
        """Valide la structure des données de ventes."""
        if self.data is None:
            self.logger.error("Aucune donnée à valider")
            return False
        
        validation_passed = True
        
        # 1. Vérifier les colonnes critiques
        critical_columns = ['date', 'product_id', 'quantity']
        for col in critical_columns:
            if col not in self.data.columns:
                self.logger.error(f"❌ Colonne critique manquante: {col}")
                validation_passed = False
        
        # 2. Vérifier les doublons sur la clé primaire
        if all(col in self.data.columns for col in self.primary_key):
            duplicates = self.data.duplicated(subset=self.primary_key).sum()
            if duplicates > 0:
                self.logger.warning(f"⚠️  {duplicates} doublons détectés sur {self.primary_key}")
                # On ne bloque pas pour ça, mais on note
        
        # 3. Vérifier les valeurs négatives pour les quantités
        if 'quantity' in self.data.columns:
            negative_qty = (self.data['quantity'] < 0).sum()
            if negative_qty > 0:
                self.logger.warning(f"⚠️  {negative_qty} quantités négatives")
        
        # 4. Vérifier les dates
        if 'date' in self.data.columns:
            null_dates = self.data['date'].isnull().sum()
            if null_dates > 0:
                self.logger.warning(f"⚠️  {null_dates} dates manquantes")
            
            # Vérifier la plage de dates
            if self.data['date'].notna().any():
                date_range = self.data['date'].dropna()
                self.logger.info(f"📅 Plage de dates: {date_range.min()} to {date_range.max()}")
        
        # 5. Statistiques de base
        if validation_passed:
            self.logger.info(f"📊 Statistiques des ventes:")
            self.logger.info(f"   - Nombre de lignes: {len(self.data)}")
            self.logger.info(f"   - Nombre de produits uniques: {self.data['product_id'].nunique()}")
            self.logger.info(f"   - Période: {self.data['date'].min()} to {self.data['date'].max()}")
            
            if 'quantity' in self.data.columns:
                self.logger.info(f"   - Quantité totale: {self.data['quantity'].sum():,.0f}")
                self.logger.info(f"   - Quantité moyenne/jour: {self.data['quantity'].mean():.2f}")
        
        return validation_passed
    
    def _update_metadata(self) -> None:
        """Met à jour les métadonnées après chargement."""
        if self.data is None:
            return
        
        self.metadata.update({
            'rows': len(self.data),
            'unique_products': self.data['product_id'].nunique() if 'product_id' in self.data.columns else 0,
            'date_min': str(self.data['date'].min()) if 'date' in self.data.columns else None,
            'date_max': str(self.data['date'].max()) if 'date' in self.data.columns else None,
            'total_quantity': float(self.data['quantity'].sum()) if 'quantity' in self.data.columns else None,
            'file_format': 'parquet' if data_config.sales_parquet.exists() else 'csv'
        })
    
    def get_daily_sales(self) -> pd.DataFrame:
        """Agrège les ventes par jour."""
        if self.data is None:
            self.logger.error("Chargez d'abord les données")
            return pd.DataFrame()
        
        if 'date' not in self.data.columns:
            self.logger.error("Colonne 'date' manquante")
            return pd.DataFrame()
        
        daily_sales = self.data.groupby('date').agg({
            'quantity': 'sum',
            'revenue': 'sum' if 'revenue' in self.data.columns else ('quantity', 'sum')
        }).reset_index()
        
        # Si revenue n'existe pas, créer une colonne factice
        if 'revenue' not in daily_sales.columns:
            daily_sales['revenue'] = daily_sales['quantity']  # À remplacer par prix réel
        
        return daily_sales
    
    def get_product_sales(self, product_id: Optional[str] = None) -> pd.DataFrame:
        """Récupère les ventes par produit ou pour un produit spécifique."""
        if self.data is None:
            return pd.DataFrame()
        
        if product_id:
            return self.data[self.data['product_id'] == product_id].copy()
        else:
            product_sales = self.data.groupby('product_id').agg({
                'quantity': 'sum',
                'date': ['min', 'max', 'nunique']
            }).reset_index()
            
            # Aplatir les colonnes multi-index
            product_sales.columns = [
                'product_id', 
                'total_quantity', 
                'first_sale', 
                'last_sale', 
                'days_with_sales'
            ]
            
            return product_sales
def _convert_dates(self, date_columns: list) -> None:
    """Convertit les colonnes de date en datetime."""
    if not date_columns:
        return
    
    for col in date_columns:
        try:
            # AVANT : self.data[col] = pd.to_datetime(self.data[col], errors='coerce', dayfirst=True)
            # APRÈS : Essayer plusieurs formats
            self.data[col] = pd.to_datetime(
                self.data[col], 
                errors='coerce',
                format='mixed',  # Détection automatique du format
                dayfirst=True    # Format français
            )
            
            # Compter les dates valides
            valid_dates = self.data[col].notna().sum()
            total_dates = len(self.data[col])
            valid_percentage = valid_dates / total_dates * 100
            
            self.logger.info(f"✅ Converti {col}: {valid_dates}/{total_dates} dates valides ({valid_percentage:.1f}%)")
            
        except Exception as e:
            self.logger.warning(f"Échec conversion de {col}: {e}")