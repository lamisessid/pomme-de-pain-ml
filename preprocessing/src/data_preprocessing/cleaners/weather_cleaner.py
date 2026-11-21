"""
Cleaner pour les données météorologiques.
"""

from typing import Optional, Dict, List
import pandas as pd
import numpy as np
import logging

from .base_cleaner import BaseCleaner
from ..config import processing_config
from ..utils import create_calendar_features


class WeatherCleaner(BaseCleaner):
    """Nettoie les données météorologiques."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.stats = {
            'invalid_values_removed': 0,
            'missing_values_filled': 0,
            'outliers_corrected': 0,
            'days_added': 0,
            'rows_processed': 0
        }
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données météorologiques.
        
        Étapes:
        1. Valider la structure
        2. S'assurer d'une entrée par jour
        3. Gérer les valeurs aberrantes
        4. Compléter les séries temporelles
        5. Créer des catégories météo
        """
        self.logger.info("🌤️  Nettoyage des données météorologiques...")
        
        if data is None or data.empty:
            self.logger.warning("⚠️  Données météo vides")
            return data
        
        data = data.copy()
        initial_rows = len(data)
        
        # 0. Renommer les colonnes au besoin (pour vos données spécifiques)
        data = self._rename_columns(data)
        
        # 1. Valider la structure
        data = self._validate_structure(data)
        
        # 2. S'assurer que 'date' est datetime et unique
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'], errors='coerce')
            data = data.sort_values('date')
            
            # Supprimer les dates invalides
            invalid_dates = data['date'].isnull().sum()
            if invalid_dates > 0:
                self.logger.warning(f"Suppression de {invalid_dates} lignes avec dates invalides")
                data = data.dropna(subset=['date'])
            
            # Supprimer les doublons de dates (garder la première)
            duplicates = data['date'].duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Suppression de {duplicates} doublons de dates")
                data = data.drop_duplicates(subset=['date'], keep='first')
        
        # 3. Compléter les dates manquantes
        data = self._complete_date_series(data)
        
        # 4. Valider les plages de valeurs
        data = self._validate_value_ranges(data)
        
        # 5. Gérer les valeurs manquantes
        data = self._handle_missing_weather(data)
        
        # 6. Gérer les outliers
        data = self._handle_weather_outliers(data)
        
        # 7. Créer des catégories météo
        data = self._create_weather_categories(data)
        
        # 8. Créer des features dérivées
        data = self._create_weather_features(data)
        
        # 9. Filtrer sur la période configurée
        data = self._filter_by_date_range(data)
        
        # Statistiques finales
        final_rows = len(data)
        self.stats['rows_processed'] = final_rows
        
        self.logger.info(f"✅ Nettoyage météo terminé: {initial_rows} → {final_rows} lignes")
        self._log_cleaning_stats()
        
        return data
    
    def _rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Renomme les colonnes au format standard."""
        column_mapping = {}
        
        for col in data.columns:
            col_lower = col.lower().strip()
            
            # Mapping spécifique pour vos données
            if col_lower in ['pluie mm', 'pluie', 'mm']:
                column_mapping[col] = 'precipitation'
            elif col_lower in ['ensoleillement', 'soleil']:
                column_mapping[col] = 'sunshine'
            elif col_lower in ['mois']:
                column_mapping[col] = 'month'
            elif col_lower in ['saison']:
                column_mapping[col] = 'season'
        
        if column_mapping:
            data = data.rename(columns=column_mapping)
            self.logger.info(f"   - Colonnes renommées: {column_mapping}")
        
        return data
    
    def _validate_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Valide la structure des données météo."""
        # Colonne date obligatoire
        if 'date' not in data.columns:
            self.logger.error("❌ Colonne 'date' manquante dans les données météo")
            raise ValueError("Colonne 'date' manquante")
        
        # S'assurer que nous avons au moins une variable météo
        weather_vars = ['temperature', 'precipitation', 'sunshine', 'humidity', 'wind_speed', 'pressure']
        existing_vars = [var for var in weather_vars if var in data.columns]
        
        if not existing_vars:
            self.logger.warning("⚠️  Aucune variable météo standard trouvée")
            # Essayer de détecter automatiquement
            data = self._detect_weather_variables(data)
            # Re-vérifier après détection
            existing_vars = [var for var in weather_vars if var in data.columns]
        
        if existing_vars:
            self.logger.info(f"   - Variables météo détectées: {existing_vars}")
        else:
            self.logger.warning("⚠️  Aucune variable météo trouvée même après détection")
        
        return data
    
    def _detect_weather_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Tente de détecter automatiquement les variables météo."""
        # Mapping de noms de colonnes communs
        var_mapping = {
            'temperature': ['temp', 't°', 'degre', 'celsius', 'temperature'],
            'precipitation': ['precip', 'pluie', 'rain', 'mm', 'pluie mm'],
            'sunshine': ['soleil', 'sun', 'ensoleillement', 'hours'],
            'humidity': ['humidite', 'humid', 'rh', '%'],
            'wind_speed': ['vent', 'wind', 'km/h', 'm/s'],
            'pressure': ['pression', 'pressure', 'hpa', 'bar']
        }
        
        renamed = False
        for col in data.columns:
            if col == 'date':
                continue
            
            col_lower = col.lower()
            for standard_var, keywords in var_mapping.items():
                if any(keyword in col_lower for keyword in keywords):
                    data = data.rename(columns={col: standard_var})
                    self.logger.info(f"     → '{col}' renommé en '{standard_var}'")
                    renamed = True
                    break
        
        return data
    
    def _complete_date_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """Complète la série temporelle pour avoir une entrée par jour."""
        if 'date' not in data.columns or len(data) == 0:
            return data
        
        # Créer une série de dates complète
        min_date = data['date'].min()
        max_date = data['date'].max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Vérifier les dates manquantes
        missing_dates = all_dates.difference(data['date'])
        
        if len(missing_dates) > 0:
            self.logger.info(f"   - {len(missing_dates)} jours manquants détectés")
            
            # Créer un DataFrame pour les dates manquantes
            missing_df = pd.DataFrame({'date': missing_dates})
            
            # Ajouter les colonnes existantes avec valeurs NaN
            for col in data.columns:
                if col != 'date':
                    missing_df[col] = np.nan
            
            # Fusionner avec les données existantes
            data = pd.concat([data, missing_df], ignore_index=True)
            data = data.sort_values('date').reset_index(drop=True)
            
            self.stats['days_added'] = len(missing_dates)
            self.logger.info(f"     → Dates manquantes ajoutées")
        
        return data
    
    def _validate_value_ranges(self, data: pd.DataFrame) -> pd.DataFrame:
        """Valide les plages de valeurs pour chaque variable météo."""
        # Plages réalistes pour SQY
        valid_ranges = {
            'temperature': (-10, 40),      # °C
            'precipitation': (0, 100),     # mm/jour
            'sunshine': (0, 24),          # heures
            'humidity': (0, 100),         # %
            'wind_speed': (0, 150),       # km/h
            'pressure': (950, 1050)       # hPa
        }
        
        for var, (min_val, max_val) in valid_ranges.items():
            # Vérifier d'abord si la colonne existe dans les données
            if var not in data.columns:
                continue
                
            # S'assurer que la colonne est numérique
            if not pd.api.types.is_numeric_dtype(data[var]):
                try:
                    # Essayer de convertir en numérique
                    data[var] = pd.to_numeric(data[var], errors='coerce')
                    self.logger.info(f"     - {var} converti en numérique")
                except Exception as e:
                    self.logger.warning(f"     - {var} ne peut pas être converti en numérique: {e}")
                    continue
            
            # Vérifier que la colonne est maintenant numérique
            if pd.api.types.is_numeric_dtype(data[var]):
                # Identifier les valeurs hors plage
                mask_out_of_range = (data[var] < min_val) | (data[var] > max_val)
                out_of_range = mask_out_of_range.sum()
                
                if out_of_range > 0:
                    self.logger.warning(f"   ⚠️  {out_of_range} valeurs de {var} hors plage [{min_val}, {max_val}]")
                    
                    # Remplacer par NaN pour interpolation plus tard
                    data.loc[mask_out_of_range, var] = np.nan
                    
                    self.stats['invalid_values_removed'] += out_of_range
            else:
                self.logger.warning(f"   ⚠️  {var} n'est pas numérique, impossible de valider la plage")
        
        return data
    
    def _handle_missing_weather(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gère les valeurs manquantes dans les données météo."""
        # Variables météo numériques
        weather_vars = [col for col in data.columns if col not in ['date', 'weather_category', 'season', 'month']]
        weather_vars = [col for col in weather_vars if pd.api.types.is_numeric_dtype(data[col])]
        
        if not weather_vars:
            return data
        
        self.logger.info("   🔍 Traitement des valeurs manquantes...")
        
        for var in weather_vars:
            missing_before = data[var].isnull().sum()
            
            if missing_before > 0:
                # Pour la météo, interpolation temporelle
                data[var] = data[var].interpolate(method='linear', limit_direction='both')
                
                missing_after = data[var].isnull().sum()
                filled = missing_before - missing_after
                
                if filled > 0:
                    self.logger.info(f"     - {var}: {filled} valeurs interpolées")
                    self.stats['missing_values_filled'] += filled
                
                # Si des valeurs manquent encore, forward/backward fill
                if missing_after > 0:
                    data[var] = data[var].ffill().bfill()
                    self.logger.info(f"     - {var}: {missing_after} valeurs forward/backward filled")
        
        # Gestion spéciale pour 'sunshine' qui pourrait être textuel
        if 'sunshine' in data.columns and not pd.api.types.is_numeric_dtype(data['sunshine']):
            self.logger.info("   🌞 Conversion de 'sunshine' textuel en numérique...")
            # Mapping des valeurs textuelles
            sunshine_map = {
                'soleil': 1.0,
                'nuageux': 0.5,
                'pluie': 0.0,
                'ensoleillé': 1.0,
                'couvert': 0.3,
                'partiellement nuageux': 0.7
            }
            
            def convert_sunshine(val):
                if pd.isna(val):
                    return 0.5  # valeur par défaut
                val_str = str(val).lower().strip()
                for key, score in sunshine_map.items():
                    if key in val_str:
                        return score
                # Essayer de convertir en numérique
                try:
                    return float(val)
                except:
                    return 0.5
            
            data['sunshine'] = data['sunshine'].apply(convert_sunshine)
            self.logger.info("     - 'sunshine' converti en scores numériques")
        
        return data
    
    def _handle_weather_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gère les outliers dans les données météo."""
        # Variables à vérifier pour les outliers (uniquement celles qui existent)
        possible_outlier_vars = ['temperature', 'precipitation', 'wind_speed']
        existing_vars = [var for var in possible_outlier_vars if var in data.columns and pd.api.types.is_numeric_dtype(data[var])]
        
        if not existing_vars:
            return data
        
        self.logger.info("   📊 Détection des outliers météo...")
        
        for var in existing_vars:
            # Vérifier qu'il y a suffisamment de données
            if len(data[var].dropna()) < 10:
                self.logger.warning(f"     - {var}: pas assez de données pour détecter les outliers")
                continue
            
            # Utiliser la méthode IQR
            Q1 = data[var].quantile(0.25)
            Q3 = data[var].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # Pas de variation
                continue
                
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identifier les outliers
            outliers_mask = (data[var] < lower_bound) | (data[var] > upper_bound)
            outliers = outliers_mask.sum()
            
            if outliers > 0:
                self.logger.info(f"     - {var}: {outliers} outliers détectés (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")
                
                # Pour la météo, on interpole plutôt que de clip
                # Marquer les outliers comme NaN pour interpolation
                data.loc[outliers_mask, var] = np.nan
                
                # Interpoler les valeurs marquées comme outliers
                data[var] = data[var].interpolate(method='linear', limit_direction='both')
                
                self.stats['outliers_corrected'] += outliers
        
        return data
    
    def _create_weather_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des catégories météorologiques."""
        # Vérifier que nous avons les colonnes nécessaires
        has_temp = 'temperature' in data.columns and pd.api.types.is_numeric_dtype(data['temperature'])
        has_precip = 'precipitation' in data.columns and pd.api.types.is_numeric_dtype(data['precipitation'])
        
        if not has_temp or not has_precip:
            self.logger.warning("   ⚠️  Impossible de créer des catégories météo: données insuffisantes")
            data['weather_category'] = 'unknown'
            return data
        
        self.logger.info("   🏷️  Création des catégories météo...")
        
        # Catégoriser la température
        def categorize_temperature(temp):
            if pd.isna(temp):
                return 'unknown'
            elif temp < 0:
                return 'very_cold'
            elif temp < 10:
                return 'cold'
            elif temp < 20:
                return 'cool'
            elif temp < 25:
                return 'mild'
            elif temp < 30:
                return 'warm'
            else:
                return 'hot'
        
        # Catégoriser les précipitations
        def categorize_precipitation(precip):
            if pd.isna(precip):
                return 'unknown'
            elif precip == 0:
                return 'dry'
            elif precip < 2.5:
                return 'light_rain'
            elif precip < 7.5:
                return 'moderate_rain'
            elif precip < 15:
                return 'heavy_rain'
            else:
                return 'very_heavy_rain'
        
        # Catégoriser globalement
        def categorize_weather(row):
            temp_cat = categorize_temperature(row.get('temperature'))
            precip_cat = categorize_precipitation(row.get('precipitation'))
            
            # Combinaison température + précipitations
            if precip_cat == 'dry':
                if temp_cat in ['very_cold', 'cold']:
                    return 'cold_dry'
                elif temp_cat in ['cool', 'mild']:
                    return 'pleasant'
                elif temp_cat in ['warm', 'hot']:
                    return 'hot_dry'
            else:
                if temp_cat in ['very_cold', 'cold']:
                    return 'cold_rainy'
                elif temp_cat in ['cool', 'mild']:
                    return 'rainy'
                elif temp_cat in ['warm', 'hot']:
                    return 'hot_rainy'
            
            return 'unknown'
        
        # Appliquer les catégories
        data['weather_category'] = data.apply(categorize_weather, axis=1)
        
        # Statistiques des catégories
        category_counts = data['weather_category'].value_counts()
        self.logger.info(f"     - Catégories créées: {dict(category_counts.head())}")
        
        return data
    
    def _create_weather_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features météorologiques dérivées."""
        if 'date' not in data.columns:
            return data
        
        # Features de tendance (moyennes mobiles) - seulement pour variables numériques
        possible_weather_vars = ['temperature', 'precipitation', 'humidity', 'sunshine']
        existing_numeric_vars = [var for var in possible_weather_vars 
                               if var in data.columns and pd.api.types.is_numeric_dtype(data[var])]
        
        for var in existing_numeric_vars:
            # Moyenne mobile sur 3, 7, 14 jours
            for window in [3, 7, 14]:
                col_name = f'{var}_ma_{window}d'
                data[col_name] = data[var].rolling(window=window, min_periods=1).mean()
            
            # Changement jour précédent
            data[f'{var}_change'] = data[var].diff()
            
            # Normalisation (z-score par mois pour saisonnalité)
            data['month_temp'] = data['date'].dt.month
            data[f'{var}_zscore'] = data.groupby('month_temp')[var].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            data = data.drop(columns=['month_temp'])
        
        # Indicateur de conditions extrêmes
        if 'temperature' in data.columns and pd.api.types.is_numeric_dtype(data['temperature']):
            data['extreme_heat'] = (data['temperature'] > 30).astype(int)
            data['extreme_cold'] = (data['temperature'] < 0).astype(int)
        
        if 'precipitation' in data.columns and pd.api.types.is_numeric_dtype(data['precipitation']):
            data['heavy_rain'] = (data['precipitation'] > 10).astype(int)
        
        # Features cycliques pour la saisonnalité
        try:
            data = create_calendar_features(data, 'date')
        except Exception as e:
            self.logger.warning(f"   ⚠️  Erreur création features calendrier: {e}")
        
        # Saison météorologique (si non déjà présente)
        if 'season' not in data.columns:
            def get_season(date):
                month = date.month
                if month in [12, 1, 2]:
                    return 'winter'
                elif month in [3, 4, 5]:
                    return 'spring'
                elif month in [6, 7, 8]:
                    return 'summer'
                else:
                    return 'autumn'
            
            data['season'] = data['date'].apply(get_season)
        
        return data
    
    def _filter_by_date_range(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filtre sur la période configurée."""
        if 'date' not in data.columns:
            return data
        
        try:
            start_date = pd.to_datetime(processing_config.start_date)
            end_date = pd.to_datetime(processing_config.end_date)
            
            before = len(data)
            data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
            after = len(data)
            
            if before != after:
                self.logger.info(f"   - Filtrage période: {before - after} lignes enlevées")
        except Exception as e:
            self.logger.warning(f"   ⚠️  Erreur lors du filtrage par période: {e}")
        
        return data
    
    def _log_cleaning_stats(self) -> None:
        """Affiche les statistiques de nettoyage."""
        self.logger.info("📊 Statistiques de nettoyage météo:")
        for key, value in self.stats.items():
            if isinstance(value, (int, float)) and value > 0:
                self.logger.info(f"   - {key}: {value}")