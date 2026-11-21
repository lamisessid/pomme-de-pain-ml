"""
Module pour fusionner toutes les sources de données.
"""

from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
import logging

from .config import data_config, processing_config
from .utils import setup_logger, save_file


class DataMerger:
    """Fusionne les données nettoyées en un dataset unique."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger(self.__class__.__name__)
        self.merge_stats = {}
    
    def merge(
        self,
        sales_data: pd.DataFrame,
        weather_data: pd.DataFrame,
        events_data: pd.DataFrame,
        products_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fusionne toutes les sources de données.
        
        Étapes:
        1. Fusion ventes × produits
        2. Ajout des données météo
        3. Ajout des données événements
        4. Création des features finales
        5. Validation du dataset fusionné
        """
        self.logger.info("🔗 Fusion des données...")
        
        # Vérifier que nous avons au moins les ventes
        if sales_data is None or sales_data.empty:
            self.logger.error("❌ Données de ventes manquantes")
            raise ValueError("Données de ventes requises pour la fusion")
        
        self.logger.info(f"   - Ventes: {sales_data.shape}")
        self.logger.info(f"   - Météo: {weather_data.shape if weather_data is not None else 'N/A'}")
        self.logger.info(f"   - Événements: {events_data.shape if events_data is not None else 'N/A'}")
        self.logger.info(f"   - Produits: {products_data.shape if products_data is not None else 'N/A'}")
        
        # 1. Fusion ventes × produits
        merged_data = self._merge_sales_products(sales_data, products_data)
        
        # 2. Ajout météo
        merged_data = self._merge_weather(merged_data, weather_data)
        
        # 3. Ajout événements
        merged_data = self._merge_events(merged_data, events_data)
        
        # 4. Création des features d'interaction
        merged_data = self._create_interaction_features(merged_data)
        
        # 5. Features de lag et rolling sur les ventes
        merged_data = self._create_sales_temporal_features(merged_data)
        
        # 6. Nettoyage final
        merged_data = self._final_cleanup(merged_data)
        
        # 7. Validation
        self._validate_merged_data(merged_data)
        
        self.logger.info(f"✅ Fusion terminée: {merged_data.shape}")
        self._log_merge_stats()
        
        return merged_data
    
    def _merge_sales_products(self, sales_data: pd.DataFrame, products_data: pd.DataFrame) -> pd.DataFrame:
        """Fusionne les données de ventes avec les informations produits."""
        self.logger.info("   1. Fusion ventes × produits...")
        
        if sales_data is None:
            return pd.DataFrame()
        
        merged = sales_data.copy()
        
        if products_data is not None and not products_data.empty:
            # Colonnes nécessaires pour la jointure
            if 'product_id' in sales_data.columns and 'product_id' in products_data.columns:
                # Vérifier le format des product_id avant la jointure
                sales_data['product_id'] = sales_data['product_id'].astype(str).str.strip()
                products_data['product_id'] = products_data['product_id'].astype(str).str.strip()
                
                # Jointure gauche pour garder toutes les ventes
                merged = pd.merge(
                    merged,
                    products_data,
                    on='product_id',
                    how='left',
                    suffixes=('', '_product')
                )
                
                # Statistiques de la jointure
                matched = merged['product_id'].isin(products_data['product_id']).sum()
                total = len(merged)
                match_rate = matched / total * 100 if total > 0 else 0
                
                self.logger.info(f"     - {matched}/{total} ventes liées à un produit ({match_rate:.1f}%)")
                self.merge_stats['sales_products_match_rate'] = match_rate
                
                # Gérer les produits non trouvés
                unmatched = merged[~merged['product_id'].isin(products_data['product_id'])]
                if not unmatched.empty:
                    self.logger.warning(f"     - {len(unmatched)} ventes sans information produit")
                    
                    # Remplir avec des valeurs par défaut pour les produits manquants
                    for col in ['category', 'name', 'sub_category']:
                        if col in products_data.columns and col in merged.columns:
                            merged[col] = merged[col].fillna('unknown')
                    for col in ['price', 'cost', 'margin_percentage']:
                        if col in products_data.columns and col in merged.columns:
                            # Utiliser la moyenne des produits existants
                            avg_val = products_data[col].mean()
                            merged[col] = merged[col].fillna(avg_val)
            else:
                missing_cols = []
                if 'product_id' not in sales_data.columns:
                    missing_cols.append('sales_data.product_id')
                if 'product_id' not in products_data.columns:
                    missing_cols.append('products_data.product_id')
                self.logger.warning(f"⚠️  Impossible de fusionner ventes × produits: {', '.join(missing_cols)} manquante(s)")
        else:
            self.logger.warning("⚠️  Données produits manquantes")
            # Créer des colonnes produits factices
            for col in ['category', 'name']:
                merged[col] = 'unknown'
            merged['price'] = merged.get('price', 1.0)
            merged['cost'] = merged.get('cost', 0.5)
        
        # Calculer le chiffre d'affaires si non présent
        if 'revenue' not in merged.columns and 'quantity' in merged.columns and 'price' in merged.columns:
            merged['revenue'] = merged['quantity'] * merged['price']
            self.logger.info("✅ Chiffre d'affaires calculé")
        
        # Calculer la marge unitaire
        if 'price' in merged.columns and 'cost' in merged.columns:
            merged['unit_margin'] = merged['price'] - merged['cost']
            merged['margin_percentage'] = (merged['unit_margin'] / merged['price'] * 100).fillna(0)
        
        self.logger.info(f"     → Shape après fusion: {merged.shape}")
        
        return merged
    
    def _merge_weather(self, data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les données météorologiques."""
        self.logger.info("   2. Ajout des données météo...")
        
        if weather_data is None or weather_data.empty:
            self.logger.warning("⚠️  Données météo manquantes, création de features par défaut")
            # Créer des features météo factices
            data['temperature'] = 15.0
            data['precipitation'] = 0.0
            data['is_rainy_day'] = 0
            return data
        
        if 'date' not in data.columns or 'date' not in weather_data.columns:
            self.logger.error("❌ Colonne 'date' manquante pour la fusion météo")
            return data
        
        # S'assurer que les dates sont au même format
        data['date'] = pd.to_datetime(data['date'])
        weather_data['date'] = pd.to_datetime(weather_data['date'])
        
        # Extraire les colonnes météo (exclure 'date' pour éviter les doublons)
        weather_cols_to_add = [col for col in weather_data.columns if col != 'date']
        
        # Jointure sur la date
        before_rows = len(data)
        data = pd.merge(
            data,
            weather_data[['date'] + weather_cols_to_add],
            on='date',
            how='left',
            suffixes=('', '_weather')
        )
        after_rows = len(data)
        
        # Vérifier la jointure
        weather_matched = data[weather_cols_to_add[0]].notna().sum() if weather_cols_to_add else 0
        match_rate = weather_matched / after_rows * 100 if after_rows > 0 else 0
        
        self.logger.info(f"     - {weather_matched}/{after_rows} lignes avec météo ({match_rate:.1f}%)")
        self.merge_stats['weather_match_rate'] = match_rate
        
        # Remplir les valeurs manquantes
        if match_rate < 100:
            self.logger.warning(f"     - {(after_rows - weather_matched)} lignes sans météo")
            
            # Interpolation temporelle pour les météo manquantes
            weather_cols = [col for col in weather_cols_to_add if col in data.columns and data[col].dtype in [np.float64, np.int64]]
            for col in weather_cols:
                if data[col].isna().any():
                    # Remplir par la moyenne du même jour de la semaine
                    data['day_of_week'] = data['date'].dt.dayofweek
                    dow_avg = data.groupby('day_of_week')[col].transform('mean')
                    data[col] = data[col].fillna(dow_avg)
                    # Nettoyer la colonne temporaire
                    if 'day_of_week' in data.columns:
                        data = data.drop(columns=['day_of_week'])
        
        self.logger.info(f"     → Colonnes météo ajoutées: {len(weather_cols_to_add)} colonnes")
        
        return data
    
    def _merge_events(self, data: pd.DataFrame, events_data: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les données d'événements."""
        self.logger.info("   3. Ajout des données événements...")
        
        if events_data is None or events_data.empty:
            self.logger.warning("⚠️  Données événements manquantes, création de features par défaut")
            # Créer des features événements factices
            data['has_event'] = 0
            data['event_count'] = 0
            data['total_impact'] = 0.0
            return data
        
        if 'date' not in data.columns or 'date' not in events_data.columns:
            self.logger.error("❌ Colonne 'date' manquante pour la fusion événements")
            return data
        
        # S'assurer que les dates sont au même format
        data['date'] = pd.to_datetime(data['date'])
        events_data['date'] = pd.to_datetime(events_data['date'])
        
        # Extraire les colonnes événements (exclure 'date' pour éviter les doublons)
        event_cols_to_add = [col for col in events_data.columns if col != 'date']
        
        # Jointure sur la date
        before_rows = len(data)
        data = pd.merge(
            data,
            events_data[['date'] + event_cols_to_add],
            on='date',
            how='left',
            suffixes=('', '_events')
        )
        after_rows = len(data)
        
        # Remplir les valeurs manquantes pour les colonnes événements
        for col in event_cols_to_add:
            if col in data.columns:
                # Pour les colonnes numériques, remplacer par 0
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].fillna(0)
                # Pour les colonnes catégorielles, remplacer par 'none'
                elif pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object':
                    data[col] = data[col].fillna('none')
        
        # Statistiques
        has_event_col = next((col for col in ['has_event', 'event_count'] if col in data.columns), None)
        if has_event_col:
            if pd.api.types.is_numeric_dtype(data[has_event_col]):
                events_matched = (data[has_event_col] > 0).sum()
            else:
                events_matched = (data[has_event_col] != 'none').sum()
        else:
            events_matched = 0
            
        self.logger.info(f"     - {events_matched} jours avec événements")
        self.merge_stats['events_matched'] = events_matched
        
        self.logger.info(f"     → Colonnes événements ajoutées: {len(event_cols_to_add)} colonnes")
        
        return data
    
    def _create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features d'interaction entre les différentes sources."""
        self.logger.info("   4. Création des features d'interaction...")
        
        # Fonction helper pour convertir en booléen/int
        def to_int_series(series):
            """Convertit une série en int (0 ou 1) pour les opérations booléennes."""
            if series.dtype in [float, int]:
                return series.fillna(0).astype(int)
            elif series.dtype == bool:
                return series.fillna(False).astype(int)
            else:
                # Essayer de convertir
                try:
                    return series.fillna(0).astype(int)
                except:
                    return (series.fillna(0) != 0).astype(int)
        
        try:
            # 1. Interactions météo × jour de semaine
            if 'temperature' in data.columns and 'day_of_week' in data.columns:
                # Température normalisée par jour de semaine
                data['temp_dow_interaction'] = data['temperature'] * (data['day_of_week'] / 6)
            
            # 2. Interactions événements × météo
            if 'has_event' in data.columns and 'temperature' in data.columns:
                # Utiliser la fonction helper
                has_event_int = to_int_series(data['has_event'])
                
                # Événement avec bonne/mauvaise météo
                data['event_good_temp'] = has_event_int * (data['temperature'] > 15).astype(int)
                data['event_bad_temp'] = has_event_int * (data['temperature'] <= 15).astype(int)
            
            # 3. Interactions événements × type de jour
            if 'has_event' in data.columns and 'is_weekend' in data.columns:
                # Utiliser la fonction helper
                has_event_int = to_int_series(data['has_event'])
                is_weekend_int = to_int_series(data['is_weekend'])
                
                data['event_weekend'] = has_event_int * is_weekend_int
                data['event_weekday'] = has_event_int * (1 - is_weekend_int)
            
            # 4. Interactions produit × météo
            if 'category' in data.columns and 'temperature' in data.columns:
                # Boissons chaudes/froides selon température
                data['hot_drink_weather'] = (
                    (data['category'] == 'drink') & (data['temperature'] < 15)
                ).astype(int)
                data['cold_drink_weather'] = (
                    (data['category'] == 'drink') & (data['temperature'] > 20)
                ).astype(int)
                
                # Salades selon température
                if 'salad' in data['category'].unique():
                    data['salad_weather'] = (
                        (data['category'] == 'salad') & (data['temperature'] > 18)
                    ).astype(int)
                
                # Soupes selon température
                if 'soup' in data['category'].unique():
                    data['soup_weather'] = (
                        (data['category'] == 'soup') & (data['temperature'] < 12)
                    ).astype(int)
            
            # 5. Interactions événements × type de produit
            if 'has_event' in data.columns and 'category' in data.columns:
                # Utiliser la fonction helper
                has_event_int = to_int_series(data['has_event'])
                
                # Sandwiches pour événements
                if 'sandwich' in data['category'].unique():
                    data['event_sandwich'] = (
                        has_event_int & (data['category'] == 'sandwich')
                    ).astype(int)
                
                # Boissons pour événements
                if 'drink' in data['category'].unique():
                    data['event_drink'] = (
                        has_event_int & (data['category'] == 'drink')
                    ).astype(int)
            
            # 6. Interactions événements × météo × produit
            if all(col in data.columns for col in ['has_event', 'is_rainy_day', 'category']):
                # Utiliser la fonction helper
                has_event_int = to_int_series(data['has_event'])
                is_rainy_int = to_int_series(data.get('is_rainy_day', pd.Series(0, index=data.index)))
                
                # Événement + pluie + boissons chaudes
                if 'drink' in data['category'].unique():
                    data['event_rainy_hot_drinks'] = (
                        has_event_int & is_rainy_int & (data['category'] == 'drink')
                    ).astype(int)
            
            # 7. Prix relatif × météo
            if 'price' in data.columns and 'temperature' in data.columns:
                # Normaliser le prix
                if data['price'].max() > data['price'].min():
                    price_norm = (data['price'] - data['price'].min()) / (data['price'].max() - data['price'].min())
                    data['price_temp_interaction'] = price_norm * (data['temperature'] / 30)
            
            # 8. Interactions météo × type de jour
            if 'temperature' in data.columns and 'is_weekend' in data.columns:
                is_weekend_int = to_int_series(data['is_weekend'])
                data['weekend_temp_effect'] = data['temperature'] * is_weekend_int
                data['weekday_temp_effect'] = data['temperature'] * (1 - is_weekend_int)
            
            self.logger.info("✅ Features d'interaction créées")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Erreur lors de la création des features d'interaction: {e}")
        
        return data
    
    def _create_sales_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features temporelles basées sur les ventes historiques."""
        self.logger.info("   5. Features temporelles des ventes...")
        
        try:
            # S'assurer que les données sont triées par date et produit
            if 'date' in data.columns and 'product_id' in data.columns:
                data = data.sort_values(['product_id', 'date'])
                
                # Features par produit
                if 'quantity' in data.columns:
                    # Lag features par produit
                    for lag in [1, 2, 3, 7, 14]:
                        col_name = f'quantity_lag_{lag}'
                        data[col_name] = data.groupby('product_id')['quantity'].shift(lag)
                    
                    # Moyennes mobiles par produit
                    for window in [3, 7, 14]:
                        mean_col = f'quantity_ma_{window}d'
                        std_col = f'quantity_std_{window}d'
                        
                        data[mean_col] = data.groupby('product_id')['quantity'].transform(
                            lambda x: x.rolling(window=window, min_periods=1).mean()
                        )
                        data[std_col] = data.groupby('product_id')['quantity'].transform(
                            lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
                        )
                    
                    # Tendances par produit
                    for window in [7, 14]:
                        trend_col = f'quantity_trend_{window}d'
                        try:
                            data[trend_col] = data.groupby('product_id')['quantity'].transform(
                                lambda x: x.rolling(window=window).apply(
                                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == window else np.nan,
                                    raw=True
                                )
                            )
                        except:
                            # Fallback si le calcul échoue
                            data[trend_col] = 0
            
            # Features agrégées par jour (tous produits confondus)
            if 'date' in data.columns and 'quantity' in data.columns:
                try:
                    daily_sales = data.groupby('date')['quantity'].sum().reset_index()
                    daily_sales = daily_sales.rename(columns={'quantity': 'daily_total_sales'})
                    
                    # Fusionner avec les données originales
                    data = pd.merge(data, daily_sales, on='date', how='left')
                    
                    # Proportion des ventes du produit par rapport au total journalier
                    data['sales_proportion'] = data['quantity'] / data['daily_total_sales'].replace(0, 1)
                except:
                    data['daily_total_sales'] = data['quantity']
                    data['sales_proportion'] = 1.0
            
            self.logger.info("✅ Features temporelles des ventes créées")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Erreur lors de la création des features temporelles: {e}")
        
        return data
    
    def _final_cleanup(self, data: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage final du dataset fusionné."""
        self.logger.info("   6. Nettoyage final...")
        
        try:
            # 1. Gérer les valeurs infinies
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in data.columns:
                    data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            
            # 2. Remplacer les NaN
            # Pour les colonnes numériques, remplacer par la médiane ou 0
            for col in numeric_cols:
                if col in data.columns and data[col].isnull().any():
                    try:
                        median_val = data[col].median(skipna=True)
                        if pd.isna(median_val):
                            median_val = 0
                        data[col] = data[col].fillna(median_val)
                    except:
                        data[col] = data[col].fillna(0)
            
            # Pour les colonnes catégorielles et objets
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if col in data.columns and data[col].isnull().any():
                    # Si c'est une colonne catégorielle
                    if pd.api.types.is_categorical_dtype(data[col]):
                        # Vérifier si 'unknown' est dans les catégories
                        if 'unknown' not in data[col].cat.categories:
                            # Ajouter 'unknown' aux catégories
                            data[col] = data[col].cat.add_categories(['unknown'])
                        # Remplir les NaN par 'unknown'
                        data[col] = data[col].fillna('unknown')
                    else:
                        # Pour les colonnes objet normales
                        # Trouver la valeur la plus fréquente
                        mode_vals = data[col].mode()
                        if not mode_vals.empty:
                            fill_value = mode_vals.iloc[0]
                        else:
                            fill_value = 'unknown'
                        data[col] = data[col].fillna(fill_value)
            
            # 3. Supprimer les colonnes avec trop de valeurs manquantes (optionnel)
            missing_threshold = 0.5  # 50%
            missing_ratio = data.isnull().sum() / len(data)
            cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index.tolist()
            
            if cols_to_drop:
                self.logger.warning(f"     - Suppression de {len(cols_to_drop)} colonnes avec >{missing_threshold*100}% de valeurs manquantes")
                data = data.drop(columns=cols_to_drop)
            
            # 4. Supprimer les doublons
            before = len(data)
            # Identifier les colonnes de données (exclure les colonnes d'index temporaires)
            data_cols = [col for col in data.columns if not col.startswith('_') and not col.endswith('_temp')]
            data = data.drop_duplicates(subset=data_cols)
            after = len(data)
            
            if before != after:
                self.logger.info(f"     - {before - after} doublons supprimés")
            
            # 5. Réinitialiser l'index
            data = data.reset_index(drop=True)
            
            # 6. Vérifier les types de données
            # Convertir les colonnes de date
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            # Convertir les colonnes catégorielles avec peu de valeurs uniques
            for col in data.columns:
                if data[col].dtype == 'object':
                    unique_count = data[col].nunique()
                    if unique_count < 50 and unique_count > 0:
                        # Ajouter 'unknown' aux catégories possibles
                        categories = list(data[col].unique())
                        if 'unknown' not in categories:
                            categories.append('unknown')
                        data[col] = pd.Categorical(data[col], categories=categories)
            
            # 7. Supprimer les lignes avec des dates invalides
            if 'date' in data.columns:
                initial_rows = len(data)
                data = data[data['date'].notna()]
                if len(data) < initial_rows:
                    self.logger.info(f"     - {initial_rows - len(data)} lignes avec dates invalides supprimées")
            
            self.logger.info(f"     ✅ Nettoyage terminé: {data.shape}")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du nettoyage final: {e}")
            raise
        
        return data
    
    def _validate_merged_data(self, data: pd.DataFrame) -> None:
        """Valide le dataset fusionné."""
        self.logger.info("   7. Validation du dataset fusionné...")
        
        validation_issues = []
        
        try:
            # 1. Vérifier la taille
            if data.empty:
                validation_issues.append("Dataset vide")
            
            # 2. Vérifier les colonnes critiques
            critical_cols = ['date', 'product_id', 'quantity']
            for col in critical_cols:
                if col not in data.columns:
                    validation_issues.append(f"Colonne critique manquante: {col}")
            
            # 3. Vérifier les plages de dates
            if 'date' in data.columns:
                min_date = data['date'].min()
                max_date = data['date'].max()
                
                self.logger.info(f"     - Période: {min_date} to {max_date}")
            
            # 4. Vérifier les valeurs aberrantes
            if 'quantity' in data.columns:
                # Vérifier les quantités négatives
                negative_qty = (data['quantity'] < 0).sum()
                if negative_qty > 0:
                    validation_issues.append(f"{negative_qty} quantités négatives")
                    # Corriger les quantités négatives
                    data.loc[data['quantity'] < 0, 'quantity'] = 0
                
                # Vérifier les quantités extrêmes
                if len(data) > 10:
                    q99 = data['quantity'].quantile(0.99)
                    extreme_qty = (data['quantity'] > q99 * 10).sum()
                    if extreme_qty > 0:
                        validation_issues.append(f"{extreme_qty} quantités extrêmes")
            
            # 5. Vérifier la complétude
            missing_total = data.isnull().sum().sum()
            total_cells = data.size
            missing_percentage = missing_total / total_cells * 100 if total_cells > 0 else 0
            
            if missing_percentage > 5:  # Plus de 5% de valeurs manquantes
                validation_issues.append(f"{missing_percentage:.1f}% de valeurs manquantes")
            
            # Log des résultats
            if validation_issues:
                self.logger.warning(f"⚠️  Problèmes de validation: {validation_issues}")
            else:
                self.logger.info("✅ Dataset validé avec succès")
            
            self.merge_stats['validation_issues'] = validation_issues
            self.merge_stats['missing_percentage'] = missing_percentage
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la validation: {e}")
            validation_issues.append(f"Erreur de validation: {e}")
    
    def _log_merge_stats(self) -> None:
        """Affiche les statistiques de fusion."""
        self.logger.info("📊 Statistiques de fusion:")
        for key, value in self.merge_stats.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"   - {key}: {value}")
            elif isinstance(value, list):
                if value:
                    self.logger.info(f"   - {key}: {len(value)} problèmes")