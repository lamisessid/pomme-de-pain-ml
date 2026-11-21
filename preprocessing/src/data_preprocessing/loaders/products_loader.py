"""
Loader pour les données produits.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import json
import logging

from .base_loader import BaseLoader
from ..config import data_config
from ..utils import load_file


class ProductsLoader(BaseLoader):
    """Charge et valide les données produits."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.expected_columns = data_config.expected_products_columns
        self.primary_key = ['product_id']
        
        # Catégories attendues (à adapter selon vos données)
        self.expected_categories = [
            'sandwich',
            'salad',
            'pastry',
            'drink',
            'dessert',
            'breakfast',
            'snack',
            'bread',
            'soup',
            'menu'
        ]
    
    def load(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Charge les données produits.
        
        Stratégie:
        1. Essayer le CSV d'abord
        2. Sinon essayer le JSON
        3. Standardiser le format
        """
        self.logger.info("Chargement des données produits...")
        
        # Déterminer le fichier source
        if file_path is None:
            # Essayer CSV d'abord
            if data_config.products_raw.exists():
                file_path = data_config.products_raw
                self.logger.info("Utilisation du fichier CSV")
            elif data_config.products_json.exists():
                file_path = data_config.products_json
                self.logger.info("Utilisation du fichier JSON")
            else:
                self.logger.error("❌ Aucun fichier produits trouvé")
                raise FileNotFoundError("Fichier produits introuvable")
        
        # Charger les données selon le format
        if file_path.suffix == '.json':
            self.data = self._load_json(file_path)
        else:
            self.data = load_file(file_path, self.logger)
        
        # Nettoyer les noms de colonnes
        self._clean_column_names()
        
        # Valider
        if self.validate():
            self.logger.info("✅ Données produits chargées avec succès")
            self._standardize_data()
            self._update_metadata()
        else:
            self.logger.warning("⚠️  Données produits chargées avec des problèmes")
        
        return self.data
    
    def _load_json(self, file_path: Path) -> pd.DataFrame:
        """Charge les données depuis un fichier JSON."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convertir en DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Si c'est un dict avec produits par ID
                if all(isinstance(v, dict) for v in data.values()):
                    df = pd.DataFrame.from_dict(data, orient='index')
                    df.index.name = 'product_id'
                    df = df.reset_index()
                else:
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame()
            
            self.logger.info(f"✅ JSON produits chargé - {len(df)} produits")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du chargement JSON produits: {e}")
            raise
    
    def _clean_column_names(self) -> None:
        """Nettoie les noms de colonnes produits."""
        if self.data is None:
            return
        
        # Mapping basé sur vos colonnes vues dans le screenshot
        column_mapping = {
    'id': 'product_id',
    'nom': 'name',
    'categorie': 'category',
    'sous_categorie': 'sub_category',
    'prix': 'price',
    'cout': 'cost',
    'duree_vie': 'shelf_life',
    'popularite': 'popularity',
    'unite': 'unit',
    'marge': 'margin',
    'marge_pourcentage': 'margin_percentage',
    'rotation_stock': 'stock_rotation',
}
        
        # Appliquer le mapping
        new_columns = []
        for col in self.data.columns:
            col_clean = col.strip()
            if col_clean in column_mapping:
                new_columns.append(column_mapping[col_clean])
            else:
                # Standardiser
                new_name = (
                    col_clean.lower()
                    .replace(' ', '_')
                    .replace('é', 'e')
                    .replace('è', 'e')
                    .replace('ê', 'e')
                    .replace('à', 'a')
                    .replace('ç', 'c')
                )
                new_columns.append(new_name)
        
        self.data.columns = new_columns
        self.logger.info(f"Colonnes produits après nettoyage: {list(self.data.columns)}")
    
    def validate(self) -> bool:
        """Valide la structure des données produits."""
        if self.data is None:
            self.logger.error("Aucune donnée à valider")
            return False
        
        validation_passed = True
        
        # 1. Vérifier la clé primaire
        if 'product_id' not in self.data.columns:
            self.logger.warning("⚠️  Colonne 'product_id' manquante, création depuis index")
            self.data['product_id'] = [f"P{str(i).zfill(3)}" for i in range(len(self.data))]
        
        # 2. Vérifier les doublons d'ID
        duplicates = self.data['product_id'].duplicated().sum()
        if duplicates > 0:
            self.logger.error(f"❌ {duplicates} doublons d'ID produit détectés")
            validation_passed = False
            # Pour l'instant, on garde le premier
            self.data = self.data.drop_duplicates(subset=['product_id'], keep='first')
        
        # 3. Vérifier les colonnes critiques
        critical_columns = ['name', 'price', 'cost']
        for col in critical_columns:
            if col not in self.data.columns:
                self.logger.warning(f"⚠️  Colonne critique manquante: {col}")
        
        # 4. Vérifier les valeurs numériques
        numeric_cols = ['price', 'cost', 'shelf_life', 'popularity', 'margin']
        for col in numeric_cols:
            if col in self.data.columns:
                # Vérifier les valeurs négatives
                negative_count = (self.data[col] < 0).sum()
                if negative_count > 0:
                    self.logger.warning(f"⚠️  {negative_count} valeurs négatives dans {col}")
                
                # Vérifier les valeurs manquantes
                missing_count = self.data[col].isnull().sum()
                if missing_count > 0:
                    self.logger.info(f"   - {col}: {missing_count} valeurs manquantes")
        
        # 5. Vérifier la colonne 'durge_vie' (spécifique à vos données)
        if 'shelf_life' in self.data.columns and self.data['shelf_life'].dtype == 'object':
            self.logger.info("Parsing spécial de la colonne shelf_life...")
            self._parse_shelf_life()
        
        # 6. Statistiques
        if validation_passed:
            self.logger.info(f"📊 Statistiques produits:")
            self.logger.info(f"   - Nombre de produits: {len(self.data)}")
            
            if 'category' in self.data.columns:
                cat_counts = self.data['category'].value_counts()
                self.logger.info(f"   - Catégories: {cat_counts.to_dict()}")
            
            if 'price' in self.data.columns:
                self.logger.info(f"   - Prix moyen: {self.data['price'].mean():.2f}€")
                self.logger.info(f"   - Fourchette prix: {self.data['price'].min():.2f}€ - {self.data['price'].max():.2f}€")
        
        return validation_passed
    
    def _parse_shelf_life(self) -> None:
        """
        Parse la colonne 'shelf_life' qui a un format spécial.
        Format vu: "2.0.85" pourrait être "jours.minimum.probabilité"
        """
        try:
            # Essayer différents parsing
            shelf_life_parsed = []
            
            for value in self.data['shelf_life']:
                if pd.isna(value):
                    shelf_life_parsed.append(None)
                    continue
                
                value_str = str(value)
                
                # Essayer de parser "2.0.85" comme jours
                if value_str.count('.') == 2:
                    parts = value_str.split('.')
                    try:
                        # Prendre la première partie comme jours
                        days = float(parts[0])
                        shelf_life_parsed.append(days)
                        continue
                    except:
                        pass
                
                # Essayer de convertir directement
                try:
                    shelf_life_parsed.append(float(value_str))
                except:
                    shelf_life_parsed.append(None)
            
            self.data['shelf_life_days'] = shelf_life_parsed
            self.logger.info(f"✅ Shelf life parsé - {self.data['shelf_life_days'].notna().sum()} valeurs")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Échec du parsing de shelf_life: {e}")
    
    def _standardize_data(self) -> None:
        """Standardise les données produits."""
        if self.data is None:
            return
        
        # 1. Standardiser les catégories
        if 'category' in self.data.columns:
            self._standardize_categories()
        
        # 2. Calculer la marge si non présente
        if 'price' in self.data.columns and 'cost' in self.data.columns:
            if 'margin' not in self.data.columns:
                self.data['margin'] = self.data['price'] - self.data['cost']
                self.logger.info("✅ Marge calculée")
            
            if 'margin_percentage' not in self.data.columns:
                self.data['margin_percentage'] = (self.data['margin'] / self.data['price'] * 100).round(2)
                self.logger.info("✅ Pourcentage de marge calculé")
        
        # 3. Standardiser les unités
        if 'unit' in self.data.columns:
            self._standardize_units()
        
        # 4. S'assurer que tous les produits ont un nom
        if 'name' not in self.data.columns:
            self.data['name'] = self.data['product_id']
        
        # 5. Créer des catégories simplifiées si besoin
        if 'category' not in self.data.columns:
            self.logger.info("Création de catégories par défaut...")
            self.data['category'] = 'other'
    
    def _standardize_categories(self) -> None:
        """Standardise les catégories de produits."""
        # Mapping des catégories
        category_mapping = {
            'sandwich': 'sandwich',
            'sandwiches': 'sandwich',
            'panini': 'sandwich',
            'baguette': 'sandwich',
            'salade': 'salad',
            'salades': 'salad',
            'pâtisserie': 'pastry',
            'patisserie': 'pastry',
            'viennoiserie': 'pastry',
            'croissant': 'pastry',
            'pain au chocolat': 'pastry',
            'boisson': 'drink',
            'boissons': 'drink',
            'jus': 'drink',
            'soda': 'drink',
            'eau': 'drink',
            'café': 'drink',
            'dessert': 'dessert',
            'desserts': 'dessert',
            'gâteau': 'dessert',
            'gateau': 'dessert',
            'petit déjeuner': 'breakfast',
            'breakfast': 'breakfast',
            'snack': 'snack',
            'encas': 'snack',
            'pain': 'bread',
            'baguette': 'bread',
            'soupe': 'soup',
            'menu': 'menu',
            'formule': 'menu',
        }
        
        self.data['category'] = (
            self.data['category']
            .astype(str)
            .str.lower()
            .str.strip()
            .map(lambda x: category_mapping.get(x, x))
        )
        
        # S'assurer que toutes les catégories sont standardisées
        unknown_cats = set(self.data['category'].unique()) - set(self.expected_categories)
        if unknown_cats:
            self.logger.info(f"Catégories non standardisées détectées: {unknown_cats}")
            # Les regrouper en 'other' ou essayer de les classifier
            for cat in unknown_cats:
                # Chercher des mots-clés
                cat_lower = cat.lower()
                for expected_cat, keywords in {
                    'sandwich': ['sand', 'panini', 'wrap'],
                    'drink': ['bois', 'jus', 'café', 'eau'],
                    'pastry': ['pâtis', 'viennois', 'croissant', 'pain au chocolat'],
                    'salad': ['salade'],
                    'dessert': ['dessert', 'gâteau'],
                }.items():
                    if any(keyword in cat_lower for keyword in keywords):
                        self.data.loc[self.data['category'] == cat, 'category'] = expected_cat
                        break
                else:
                    self.data.loc[self.data['category'] == cat, 'category'] = 'other'
        
        self.logger.info(f"Catégories standardisées: {self.data['category'].value_counts().to_dict()}")
    
    def _standardize_units(self) -> None:
        """Standardise les unités de mesure."""
        unit_mapping = {
            'pièce': 'piece',
            'piece': 'piece',
            'pc': 'piece',
            'unité': 'piece',
            'unit': 'piece',
            'portion': 'portion',
            'part': 'portion',
            'serving': 'portion',
            'tasse': 'cup',
            'cup': 'cup',
            'verre': 'glass',
            'glass': 'glass',
            'bouteille': 'bottle',
            'bottle': 'bottle',
            'litre': 'liter',
            'l': 'liter',
            'gramme': 'gram',
            'g': 'gram',
            'kg': 'kilogram',
        }
        
        self.data['unit'] = (
            self.data['unit']
            .astype(str)
            .str.lower()
            .str.strip()
            .map(lambda x: unit_mapping.get(x, x))
        )
    
    def _update_metadata(self) -> None:
        """Met à jour les métadonnées."""
        if self.data is None:
            return
        
        self.metadata.update({
            'product_count': len(self.data),
            'categories': self.data['category'].value_counts().to_dict() if 'category' in self.data.columns else {},
            'avg_price': float(self.data['price'].mean()) if 'price' in self.data.columns else None,
            'avg_margin': float(self.data['margin'].mean()) if 'margin' in self.data.columns else None,
            'unique_units': list(self.data['unit'].unique()) if 'unit' in self.data.columns else []
        })
    
    def get_product_by_id(self, product_id: str) -> Dict[str, Any]:
        """Récupère les informations d'un produit par son ID."""
        if self.data is None or product_id not in self.data['product_id'].values:
            return {}
        
        product_row = self.data[self.data['product_id'] == product_id].iloc[0]
        return product_row.to_dict()
    
    def get_products_by_category(self, category: str) -> pd.DataFrame:
        """Récupère tous les produits d'une catégorie."""
        if self.data is None or 'category' not in self.data.columns:
            return pd.DataFrame()
        
        return self.data[self.data['category'] == category].copy()