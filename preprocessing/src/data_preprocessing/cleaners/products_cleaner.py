"""
Cleaner pour les données produits.
"""

from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import logging

from .base_cleaner import BaseCleaner
from ..config import processing_config


class ProductsCleaner(BaseCleaner):
    """Nettoie les données produits."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.stats = {
            'products_processed': 0,
            'invalid_prices_fixed': 0,
            'categories_standardized': 0,
            'missing_values_filled': 0
        }
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données produits.
        
        Étapes:
        1. Valider la structure
        2. Standardiser les IDs produits
        3. Valider et corriger les prix
        4. Standardiser les catégories
        5. Calculer les marges
        6. Créer des features produit
        """
        self.logger.info("🛒 Nettoyage des données produits...")
        
        if data is None or data.empty:
            self.logger.error("❌ Données produits vides")
            raise ValueError("Données produits vides")
        
        data = data.copy()
        initial_products = len(data)
        
        # 1. Valider la structure
        data = self._validate_structure(data)
        
        # 2. Standardiser les IDs
        data = self._standardize_product_ids(data)
        
        # 3. Valider et corriger les prix
        data = self._validate_prices(data)
        
        # 4. Standardiser les catégories
        data = self._standardize_categories(data)
        
        # 5. Calculer les marges si nécessaire
        data = self._calculate_margins(data)
        
        # 6. Gérer la durée de vie
        data = self._handle_shelf_life(data)
        
        # 7. Créer des features produit
        data = self._create_product_features(data)
        
        # 8. Gérer les valeurs manquantes
        data = self._handle_missing_values(data)
        
        # Statistiques finales
        final_products = len(data)
        self.stats['products_processed'] = final_products
        
        self.logger.info(f"✅ Nettoyage produits terminé: {initial_products} → {final_products} produits")
        self._log_cleaning_stats()
        
        return data
    
    def _validate_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Valide la structure des données produits."""
        # Vérifier la colonne product_id
        if 'product_id' not in data.columns:
            # Essayer de trouver un identifiant
            id_candidates = ['id', 'ID', 'identifiant', 'code', 'sku']
            for candidate in id_candidates:
                if candidate in data.columns:
                    data = data.rename(columns={candidate: 'product_id'})
                    self.logger.info(f"   → Colonne '{candidate}' renommée en 'product_id'")
                    break
            else:
                # Créer un ID depuis l'index
                self.logger.warning("⚠️  Colonne 'product_id' manquante, création depuis index")
                data['product_id'] = [f"P{str(i).zfill(3)}" for i in range(len(data))]
        
        # S'assurer que product_id est string et unique
        data['product_id'] = data['product_id'].astype(str).str.strip()
        
        # Supprimer les doublons
        duplicates = data['product_id'].duplicated().sum()
        if duplicates > 0:
            self.logger.warning(f"⚠️  {duplicates} doublons d'ID produit détectés")
            data = data.drop_duplicates(subset=['product_id'], keep='first')
        
        # Créer un nom si manquant
        if 'name' not in data.columns:
            self.logger.warning("⚠️  Colonne 'name' manquante, création depuis product_id")
            data['name'] = data['product_id']
        
        return data
    
    def _standardize_product_ids(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardise les IDs produits."""
        # Nettoyer les IDs
        data['product_id'] = (
            data['product_id']
            .astype(str)
            .str.upper()  # Tout en majuscules
            .str.strip()
            .str.replace(' ', '_')
            .str.replace('-', '_')
        )
        
        # S'assurer que les IDs commencent par une lettre
        def format_product_id(pid):
            if pid and not pid[0].isalpha():
                return f"P{pid}"
            return pid
        
        data['product_id'] = data['product_id'].apply(format_product_id)
        
        self.logger.info(f"   - IDs produits standardisés")
        self.logger.info(f"     Exemple: {data['product_id'].head(5).tolist()}")
        
        return data
    
    def _validate_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Valide et corrige les prix et coûts."""
        # Colonnes de prix attendues
        price_cols = ['price', 'cost']
        
        for col in price_cols:
            if col in data.columns:
                # Convertir en numérique
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Vérifier les valeurs négatives
                negative_count = (data[col] < 0).sum()
                if negative_count > 0:
                    self.logger.warning(f"⚠️  {negative_count} prix/couts négatifs dans {col}")
                    # Remplacer par NaN pour traitement
                    data.loc[data[col] < 0, col] = np.nan
                    self.stats['invalid_prices_fixed'] += negative_count
                
                # Vérifier les valeurs aberrantes (prix > 100€ pour un fast-food)
                if col == 'price':
                    outliers = (data[col] > 100).sum()
                    if outliers > 0:
                        self.logger.warning(f"⚠️  {outliers} prix supérieurs à 100€")
                        # Limiter à 100€
                        data.loc[data[col] > 100, col] = 100
                
                # Statistiques
                self.logger.info(f"   - {col}: moy={data[col].mean():.2f}€, min={data[col].min():.2f}€, max={data[col].max():.2f}€")
        
        # Vérifier la cohérence prix > coût
        if 'price' in data.columns and 'cost' in data.columns:
            inconsistent = (data['price'] < data['cost']).sum()
            if inconsistent > 0:
                self.logger.warning(f"⚠️  {inconsistent} produits avec prix < coût")
                # Ajuster le prix pour avoir une marge minimale de 10%
                mask = data['price'] < data['cost']
                data.loc[mask, 'price'] = data.loc[mask, 'cost'] * 1.1
        
        return data
    
    def _standardize_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardise les catégories de produits."""
        if 'category' not in data.columns:
            self.logger.warning("⚠️  Colonne 'category' manquante, création automatique")
            data['category'] = 'unknown'
            return data
        
        # Catégories standard pour Pomme de Pain
        standard_categories = {
            # Sandwiches & Baguettes
            'sandwich': ['sandwich', 'panini', 'wrap', 'burger', 'sandwiches'],
            'baguette': ['baguette', 'tradition', 'pain', 'baguettes'],
            
            # Salades
            'salad': ['salade', 'salades', 'salad', 'salads'],
            
            # Pâtisseries & Viennoiseries
            'pastry': ['pâtisserie', 'patisserie', 'viennoiserie', 'croissant',
                      'pain_au_chocolat', 'chocolatine', 'brioche'],
            
            # Boissons
            'drink': ['boisson', 'boissons', 'drink', 'drinks', 'café',
                     'the', 'jus', 'soda', 'eau', 'coca'],
            
            # Desserts
            'dessert': ['dessert', 'desserts', 'gateau', 'gâteau', 'tarte',
                       'muffin', 'cookie', 'brownie'],
            
            # Petit-déjeuner
            'breakfast': ['petit_dejeuner', 'breakfast', 'dejeuner'],
            
            # Soupes
            'soup': ['soupe', 'soupes', 'soup'],
            
            # Menus
            'menu': ['menu', 'formule', 'combo', 'menus'],
            
            # Divers
            'snack': ['snack', 'encas', 'goûter'],
            'other': ['autre', 'other', 'divers', 'unknown']
        }
        
        # Fonction de mapping
        def map_to_standard(cat):
            if not isinstance(cat, str):
                return 'other'
            
            cat_lower = cat.lower().strip()
            
            # Chercher dans les catégories standard
            for standard_cat, keywords in standard_categories.items():
                if cat_lower in keywords:
                    return standard_cat
            
            # Chercher par mots-clés
            for standard_cat, keywords in standard_categories.items():
                for keyword in keywords:
                    if keyword in cat_lower:
                        return standard_cat
            
            # Vérifier les catégories courantes
            if 'sandwich' in cat_lower:
                return 'sandwich'
            elif 'salad' in cat_lower or 'salade' in cat_lower:
                return 'salad'
            elif 'drink' in cat_lower or 'boisson' in cat_lower:
                return 'drink'
            elif 'dessert' in cat_lower:
                return 'dessert'
            elif 'pastry' in cat_lower or 'viennois' in cat_lower:
                return 'pastry'
            elif 'breakfast' in cat_lower or 'petit' in cat_lower:
                return 'breakfast'
            elif 'soup' in cat_lower or 'soupe' in cat_lower:
                return 'soup'
            elif 'menu' in cat_lower:
                return 'menu'
            else:
                return 'other'
        
        # Appliquer le mapping
        original_categories = data['category'].unique()
        data['category'] = data['category'].apply(map_to_standard)
        
        # Statistiques
        new_categories = data['category'].value_counts()
        self.logger.info(f"   - Catégories standardisées: {new_categories.to_dict()}")
        
        self.stats['categories_standardized'] = len(original_categories)
        
        return data
    
    def _calculate_margins(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule les marges si non fournies."""
        # Vérifier si les colonnes de marge existent
        has_margin = 'margin' in data.columns
        has_margin_pct = 'margin_percentage' in data.columns
        
        # Si on a prix et coût, on peut calculer les marges
        if 'price' in data.columns and 'cost' in data.columns:
            if not has_margin:
                data['margin'] = data['price'] - data['cost']
                self.logger.info("✅ Marge calculée")
            
            if not has_margin_pct:
                # Éviter la division par zéro
                with np.errstate(divide='ignore', invalid='ignore'):
                    data['margin_percentage'] = np.where(
                        data['price'] > 0,
                        (data['margin'] / data['price']) * 100,
                        0
                    )
                self.logger.info("✅ Pourcentage de marge calculé")
        
        # Vérifier la cohérence des marges
        if 'margin_percentage' in data.columns:
            # Marges négatives ou trop élevées
            invalid_margins = (
                (data['margin_percentage'] < 0) | 
                (data['margin_percentage'] > 100)
            ).sum()
            
            if invalid_margins > 0:
                self.logger.warning(f"⚠️  {invalid_margins} marges incohérentes")
                # Corriger les marges négatives
                data.loc[data['margin_percentage'] < 0, 'margin_percentage'] = 10  # Marge minimale de 10%
                # Limiter à 100%
                data['margin_percentage'] = data['margin_percentage'].clip(0, 100)
        
        # Statistiques de marge
        if 'margin_percentage' in data.columns:
            avg_margin = data['margin_percentage'].mean()
            self.logger.info(f"   - Marge moyenne: {avg_margin:.1f}%")
            
            # Marge par catégorie
            if 'category' in data.columns:
                margin_by_cat = data.groupby('category')['margin_percentage'].mean()
                self.logger.info(f"   - Marge par catégorie: {margin_by_cat.round(1).to_dict()}")
        
        return data
    
    def _handle_shelf_life(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gère la durée de vie des produits."""
        shelf_life_cols = [col for col in data.columns if 'shelf' in col.lower() or 'vie' in col.lower()]
        
        if shelf_life_cols:
            main_col = shelf_life_cols[0]
            
            # Convertir en numérique
            data['shelf_life_days'] = pd.to_numeric(data[main_col], errors='coerce')
            
            # Remplacer les valeurs manquantes par la médiane par catégorie
            if 'category' in data.columns and data['shelf_life_days'].isnull().any():
                for category in data['category'].unique():
                    mask = data['category'] == category
                    median_val = data.loc[mask, 'shelf_life_days'].median()
                    if pd.notna(median_val):
                        data.loc[mask & data['shelf_life_days'].isna(), 'shelf_life_days'] = median_val
                
                # Remplacer les restants avec la médiane globale
                global_median = data['shelf_life_days'].median()
                data['shelf_life_days'] = data['shelf_life_days'].fillna(global_median)
            
            # Catégoriser la durée de vie
            def categorize_shelf_life(days):
                if pd.isna(days):
                    return 'unknown'
                elif days < 1:
                    return 'very_short'  # Moins d'un jour
                elif days <= 2:
                    return 'short'       # 1-2 jours
                elif days <= 7:
                    return 'medium'      # 3-7 jours
                else:
                    return 'long'        # Plus d'une semaine
            
            data['shelf_life_category'] = data['shelf_life_days'].apply(categorize_shelf_life)
            
            self.logger.info(f"   - Durée de vie: moy={data['shelf_life_days'].mean():.1f} jours")
            self.logger.info(f"   - Catégories: {data['shelf_life_category'].value_counts().to_dict()}")
        
        return data
    
    def _create_product_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features pour les produits."""
        # Indicateur de produit populaire (si colonne popularity existe)
        if 'popularity' in data.columns:
            # Normaliser la popularité (0-1)
            if data['popularity'].max() > data['popularity'].min():
                data['popularity_norm'] = (
                    (data['popularity'] - data['popularity'].min()) / 
                    (data['popularity'].max() - data['popularity'].min())
                )
            else:
                data['popularity_norm'] = 0.5
            
            # Produit très populaire (top 20%)
            popularity_threshold = data['popularity'].quantile(0.8)
            data['is_very_popular'] = (data['popularity'] >= popularity_threshold).astype(int)
        
        # Indicateur de haute marge (marge > 60%)
        if 'margin_percentage' in data.columns:
            data['is_high_margin'] = (data['margin_percentage'] > 60).astype(int)
        
        # Indicateur de bas prix (prix < 3€)
        if 'price' in data.columns:
            data['is_low_price'] = (data['price'] < 3).astype(int)
            data['is_high_price'] = (data['price'] > 6).astype(int)
        
        # Catégorie de prix
        if 'price' in data.columns:
            bins = [0, 2, 4, 6, 10, float('inf')]
            labels = ['very_low', 'low', 'medium', 'high', 'very_high']
            data['price_category'] = pd.cut(data['price'], bins=bins, labels=labels)
        
        # Encodage one-hot pour les catégories (optionnel, fait plus tard)
        # data = pd.get_dummies(data, columns=['category'], prefix='cat')
        
        # Ratio prix/coût
        if 'price' in data.columns and 'cost' in data.columns:
            data['price_cost_ratio'] = data['price'] / data['cost'].replace(0, np.nan)
            data['price_cost_ratio'] = data['price_cost_ratio'].fillna(1)
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gère les valeurs manquantes dans les données produits."""
        # Colonnes numériques
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            missing_before = data[col].isnull().sum()
            
            if missing_before > 0:
                # Remplacer par la médiane de la catégorie si possible
                if 'category' in data.columns:
                    for category in data['category'].unique():
                        mask = data['category'] == category
                        median_val = data.loc[mask, col].median()
                        if pd.notna(median_val):
                            data.loc[mask & data[col].isna(), col] = median_val
                
                # Remplacer les restants par la médiane globale
                global_median = data[col].median()
                data[col] = data[col].fillna(global_median)
                
                missing_after = data[col].isnull().sum()
                filled = missing_before - missing_after
                
                if filled > 0:
                    self.logger.info(f"   - {col}: {filled} valeurs manquantes remplies")
                    self.stats['missing_values_filled'] += filled
        
        # Colonnes catégorielles
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col == 'product_id':  # Ne pas toucher à l'ID
                continue
            
            missing = data[col].isnull().sum()
            if missing > 0:
                # Remplacer par la valeur la plus fréquente
                most_common = data[col].mode()
                if not most_common.empty:
                    data[col] = data[col].fillna(most_common.iloc[0])
                    self.logger.info(f"   - {col}: {missing} valeurs manquantes remplies")
        
        return data
    
    def _log_cleaning_stats(self) -> None:
        """Affiche les statistiques de nettoyage."""
        self.logger.info("📊 Statistiques de nettoyage produits:")
        for key, value in self.stats.items():
            if isinstance(value, (int, float)) and value > 0:
                self.logger.info(f"   - {key}: {value}")