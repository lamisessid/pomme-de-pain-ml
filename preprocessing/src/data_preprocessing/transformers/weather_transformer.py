"""
Transformer pour les features météorologiques.
"""

from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import logging


class WeatherTransformer:
    """Crée des features météorologiques avancées."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.features_created = []
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des features météorologiques avancées.
        
        Features créées:
        - Catégories météo combinées
        - Indices de confort
        - Tendances météo
        - Features d'interaction avec temporal
        """
        self.logger.info("🌤️  Création des features météorologiques avancées...")
        
        if data is None or data.empty:
            self.logger.warning("⚠️  Données météo vides")
            return data
        
        if 'date' not in data.columns:
            self.logger.error("❌ Colonne 'date' non trouvée")
            raise ValueError("Colonne 'date' non trouvée")
        
        data = data.copy()
        
        # 1. Catégories météo détaillées
        data = self._create_detailed_weather_categories(data)
        
        # 2. Indices de confort et météo ressentie
        data = self._create_comfort_indices(data)
        
        # 3. Features de tendance météo
        data = self._create_weather_trends(data)
        
        # 4. Features d'interaction météo × temporal
        data = self._create_weather_temporal_interactions(data)
        
        # 5. Features d'impact météo sur ventes (si disponibles)
        data = self._create_weather_impact_features(data)
        
        # 6. Aggrégations météo par période
        data = self._create_weather_aggregations(data)
        
        self.logger.info(f"✅ {len(self.features_created)} features météo créées")
        
        return data
    
    def _create_detailed_weather_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des catégories météorologiques détaillées."""
        self.logger.info("   🏷️  Catégories météo détaillées...")
        
        # 1. Catégorisation température
        if 'temperature' in data.columns:
            # Catégories de température détaillées
            temp_bins = [-float('inf'), 0, 5, 10, 15, 20, 25, 30, float('inf')]
            temp_labels = [
                'extreme_cold', 'very_cold', 'cold', 'cool', 
                'mild', 'warm', 'hot', 'extreme_hot'
            ]
            
            data['temp_category'] = pd.cut(
                data['temperature'], 
                bins=temp_bins, 
                labels=temp_labels
            )
            
            # Température ressentie (wind chill / heat index)
            data['feels_like_temp'] = self._calculate_feels_like_temp(data)
            
            self.features_created.extend(['temp_category', 'feels_like_temp'])
        
        # 2. Catégorisation précipitations
        if 'precipitation' in data.columns:
            # Intensité de la pluie
            rain_bins = [-0.1, 0.1, 2.5, 7.5, 15, 30, float('inf')]
            rain_labels = ['dry', 'drizzle', 'light_rain', 'moderate_rain', 
                          'heavy_rain', 'violent_rain']
            
            data['rain_intensity'] = pd.cut(
                data['precipitation'], 
                bins=rain_bins, 
                labels=rain_labels
            )
            
            # Indicateur de jour de pluie
            data['is_rainy_day'] = (data['precipitation'] > 0.1).astype(int)
            
            # Séquences de jours de pluie
            data['rain_streak'] = self._calculate_streak(data['is_rainy_day'])
            
            self.features_created.extend(['rain_intensity', 'is_rainy_day', 'rain_streak'])
        
        # 3. Catégorisation ensoleillement
        if 'sunshine' in data.columns:
            # Heures d'ensoleillement normalisées (pour la saison)
            data['sunshine_norm'] = data['sunshine'] / data['sunshine'].max() if data['sunshine'].max() > 0 else 0
            
            # Jour ensoleillé vs couvert
            data['is_sunny_day'] = (data['sunshine'] > 6).astype(int)
            data['is_overcast_day'] = (data['sunshine'] < 2).astype(int)
            
            self.features_created.extend(['sunshine_norm', 'is_sunny_day', 'is_overcast_day'])
        
        # 4. Catégorisation humidité
        if 'humidity' in data.columns:
            # Niveaux d'humidité
            humidity_bins = [0, 30, 50, 70, 90, 100]
            humidity_labels = ['very_dry', 'dry', 'comfortable', 'humid', 'very_humid']
            
            data['humidity_level'] = pd.cut(
                data['humidity'], 
                bins=humidity_bins, 
                labels=humidity_labels
            )
            
            self.features_created.append('humidity_level')
        
        # 5. Catégorisation combinée météo
        data = self._create_combined_weather_category(data)
        
        return data
    
    def _calculate_feels_like_temp(self, data: pd.DataFrame) -> pd.Series:
        """Calcule la température ressentie (indice de chaleur/refroidissement éolien)."""
        # Formule simplifiée de l'indice de chaleur (Heat Index)
        # et du refroidissement éolien (Wind Chill)
        
        feels_like = data['temperature'].copy()
        
        # Si humidité disponible, calculer heat index pour temp > 20°C
        if 'humidity' in data.columns:
            mask_heat = data['temperature'] > 20
            if mask_heat.any():
                # Formule simplifiée de l'indice de chaleur
                T = data.loc[mask_heat, 'temperature']
                RH = data.loc[mask_heat, 'humidity']
                
                # Formule approximative
                heat_index = (
                    0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (RH * 0.094))
                )
                feels_like[mask_heat] = heat_index
        
        # Si vitesse vent disponible, calculer wind chill pour temp < 10°C
        if 'wind_speed' in data.columns:
            mask_cold = data['temperature'] < 10
            if mask_cold.any():
                T = data.loc[mask_cold, 'temperature']
                V = data.loc[mask_cold, 'wind_speed']
                
                # Formule simplifiée de refroidissement éolien
                wind_chill = 13.12 + 0.6215 * T - 11.37 * (V ** 0.16) + 0.3965 * T * (V ** 0.16)
                feels_like[mask_cold] = wind_chill
        
        return feels_like
    
    def _calculate_streak(self, series: pd.Series) -> pd.Series:
        """Calcule les séquences consécutives (ex: jours de pluie consécutifs)."""
        streak = series.copy()
        current_streak = 0
        
        for i in range(len(series)):
            if series.iloc[i] == 1:
                current_streak += 1
            else:
                current_streak = 0
            streak.iloc[i] = current_streak
        
        return streak
    
    def _create_combined_weather_category(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée une catégorie météo combinée."""
        self.logger.info("   🌈 Catégorie météo combinée...")
        
        def get_combined_category(row):
            # Basé sur température et précipitations
            temp = row.get('temperature', 20)
            precip = row.get('precipitation', 0)
            
            if pd.isna(temp) or pd.isna(precip):
                return 'unknown'
            
            # Catégorisation combinée
            if temp < 0:
                if precip > 0:
                    return 'cold_snowy'
                else:
                    return 'cold_dry'
            elif temp < 10:
                if precip > 2.5:
                    return 'cool_rainy'
                elif precip > 0:
                    return 'cool_drizzly'
                else:
                    return 'cool_dry'
            elif temp < 20:
                if precip > 2.5:
                    return 'mild_rainy'
                elif precip > 0:
                    return 'mild_drizzly'
                else:
                    return 'mild_dry'
            elif temp < 30:
                if precip > 2.5:
                    return 'warm_rainy'
                elif precip > 0:
                    return 'warm_drizzly'
                else:
                    return 'warm_dry'
            else:
                if precip > 0:
                    return 'hot_rainy'
                else:
                    return 'hot_dry'
        
        data['weather_combined'] = data.apply(get_combined_category, axis=1)
        self.features_created.append('weather_combined')
        
        # Statistiques
        cat_counts = data['weather_combined'].value_counts()
        self.logger.info(f"     - Distribution: {cat_counts.head(5).to_dict()}")
        
        return data
    
    def _create_comfort_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des indices de confort météorologique."""
        self.logger.info("   😌 Indices de confort...")
        
        # 1. Indice de confort thermique
        if 'temperature' in data.columns and 'humidity' in data.columns:
            # Formule simplifiée de l'indice de confort
            T = data['temperature']
            H = data['humidity']
            
            # Indice de confort (0-100, plus élevé = plus confortable)
            data['comfort_index'] = (
                100 - 
                abs(T - 22) * 2 -  # Température idéale = 22°C
                abs(H - 50) * 0.5   # Humidité idéale = 50%
            )
            
            # Borne entre 0 et 100
            data['comfort_index'] = data['comfort_index'].clip(0, 100)
            
            # Catégories de confort
            data['comfort_level'] = pd.cut(
                data['comfort_index'],
                bins=[0, 40, 60, 80, 100],
                labels=['uncomfortable', 'acceptable', 'comfortable', 'very_comfortable']
            )
            
            self.features_created.extend(['comfort_index', 'comfort_level'])
        
        # 2. Indice d'agrément extérieur
        if all(col in data.columns for col in ['temperature', 'precipitation', 'wind_speed']):
            # Facteurs négatifs
            rain_penalty = np.where(data['precipitation'] > 0, 30, 0)
            wind_penalty = data['wind_speed'] * 2
            temp_penalty = abs(data['temperature'] - 20) * 3
            
            data['outdoor_pleasure_index'] = (
                100 - rain_penalty - wind_penalty - temp_penalty
            ).clip(0, 100)
            
            # Bon pour les activités extérieures?
            data['good_for_outdoor'] = (data['outdoor_pleasure_index'] > 60).astype(int)
            
            self.features_created.extend(['outdoor_pleasure_index', 'good_for_outdoor'])
        
        # 3. Indice météo pour la vente (hypothèse: beau temps = + ventes)
        if 'temperature' in data.columns:
            # Température idéale pour les ventes de fast-food?
            # Hypothèse: entre 15 et 25°C c'est bon
            temp_score = np.where(
                (data['temperature'] >= 15) & (data['temperature'] <= 25),
                1.0,
                0.5  # Pénalité pour températures extrêmes
            )
            
            # Pluie réduit les ventes
            rain_score = np.where(data['precipitation'] > 0, 0.7, 1.0)
            
            data['weather_sales_score'] = temp_score * rain_score
            
            self.features_created.append('weather_sales_score')
        
        return data
    
    def _create_weather_trends(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features de tendance météorologique."""
        self.logger.info("   📈 Tendances météo...")
        
        # Variables météo à analyser
        weather_vars = ['temperature', 'precipitation', 'humidity']
        existing_vars = [var for var in weather_vars if var in data.columns]
        
        for var in existing_vars:
            # Changement jour précédent
            data[f'{var}_change'] = data[var].diff()
            self.features_created.append(f'{var}_change')
            
            # Changement sur 3 jours
            data[f'{var}_change_3d'] = data[var].diff(3)
            self.features_created.append(f'{var}_change_3d')
            
            # Moyenne mobile et tendance
            for window in [3, 7]:
                # Moyenne mobile
                ma_col = f'{var}_ma_{window}d'
                data[ma_col] = data[var].rolling(window=window, min_periods=1).mean()
                self.features_created.append(ma_col)
                
                # Tendance (pente de régression linéaire)
                trend_col = f'{var}_trend_{window}d'
                data[trend_col] = data[var].rolling(window=window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan,
                    raw=True
                )
                self.features_created.append(trend_col)
            
            # Volatilité (écart-type mobile)
            std_col = f'{var}_volatility_7d'
            data[std_col] = data[var].rolling(window=7, min_periods=1).std()
            self.features_created.append(std_col)
        
        # Changement de temps (ex: passage de sec à pluvieux)
        if 'precipitation' in data.columns:
            data['weather_change_type'] = 'stable'
            
            # De sec à pluvieux
            mask_dry_to_wet = (
                (data['precipitation'].shift(1) <= 0.1) & 
                (data['precipitation'] > 2.5)
            )
            data.loc[mask_dry_to_wet, 'weather_change_type'] = 'dry_to_wet'
            
            # De pluvieux à sec
            mask_wet_to_dry = (
                (data['precipitation'].shift(1) > 2.5) & 
                (data['precipitation'] <= 0.1)
            )
            data.loc[mask_wet_to_dry, 'weather_change_type'] = 'wet_to_dry'
            
            self.features_created.append('weather_change_type')
        
        return data
    
    def _create_weather_temporal_interactions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features d'interaction météo × temporal."""
        self.logger.info("   🔄 Interactions météo-temporelles...")
        
        # 1. Effet saisonnier sur la température
        if 'temperature' in data.columns and 'month' in data.columns:
            # Température normalisée par rapport à la moyenne mensuelle
            monthly_avg = data.groupby('month')['temperature'].transform('mean')
            monthly_std = data.groupby('month')['temperature'].transform('std')
            
            data['temp_zscore_monthly'] = (
                (data['temperature'] - monthly_avg) / monthly_std.replace(0, 1)
            )
            
            # Anomalie de température (écart à la normale saisonnière)
            data['temp_anomaly'] = data['temperature'] - monthly_avg
            
            self.features_created.extend(['temp_zscore_monthly', 'temp_anomaly'])
        
        # 2. Impact météo différencié par jour de semaine
        if 'temperature' in data.columns and 'day_of_week' in data.columns:
            # Température moyenne par jour de semaine
            dow_avg = data.groupby('day_of_week')['temperature'].transform('mean')
            data['temp_vs_dow_avg'] = data['temperature'] - dow_avg
            
            self.features_created.append('temp_vs_dow_avg')
        
        # 3. Météo du week-end vs semaine
        if 'is_weekend' in data.columns:
            for var in ['temperature', 'precipitation']:
                if var in data.columns:
                    col_name = f'{var}_weekend_effect'
                    # Différence entre météo réelle et moyenne de la semaine
                    weekday_avg = data[~data['is_weekend'].astype(bool)][var].mean()
                    data[col_name] = data[var] - weekday_avg
                    self.features_created.append(col_name)
        
        # 4. Météo pendant les vacances vs jours normaux
        if 'is_school_holiday' in data.columns:
            for var in ['temperature', 'precipitation']:
                if var in data.columns:
                    col_name = f'{var}_holiday_effect'
                    # Différence pendant les vacances
                    holiday_avg = data[data['is_school_holiday'] == 1][var].mean()
                    data[col_name] = data[var] - holiday_avg
                    self.features_created.append(col_name)
        
        return data
    
    def _create_weather_impact_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features d'impact météo sur le comportement client."""
        self.logger.info("   🛒 Impact météo sur comportement...")
        
        # 1. Indicateur de "météo à emporter" vs "météo sur place"
        if 'temperature' in data.columns and 'precipitation' in data.columns:
            # Météo favorable pour manger sur place (terrasse)
            data['good_for_dine_in'] = (
                (data['temperature'] >= 15) & 
                (data['temperature'] <= 28) & 
                (data['precipitation'] < 0.5) &
                (data.get('wind_speed', 0) < 20)
            ).astype(int)
            
            # Météo favorable pour emporter (mauvais temps)
            data['good_for_takeaway'] = (
                (data['precipitation'] > 2) | 
                (data['temperature'] < 5) | 
                (data['temperature'] > 30)
            ).astype(int)
            
            self.features_created.extend(['good_for_dine_in', 'good_for_takeaway'])
        
        # 2. Indicateur de météo "confort café"
        if 'temperature' in data.columns:
            # Bonne météo pour consommer des boissons chaudes
            data['good_for_hot_drinks'] = (
                (data['temperature'] < 15) | 
                (data['precipitation'] > 1)
            ).astype(int)
            
            # Bonne météo pour boissons froides
            data['good_for_cold_drinks'] = (data['temperature'] > 20).astype(int)
            
            self.features_created.extend(['good_for_hot_drinks', 'good_for_cold_drinks'])
        
        # 3. Indicateur de météo "confort salade"
        if 'temperature' in data.columns:
            # Les salades se vendent mieux quand il fait chaud
            data['good_for_salads'] = (data['temperature'] > 18).astype(int)
            
            # Les soupes se vendent mieux quand il fait froid
            data['good_for_soups'] = (data['temperature'] < 12).astype(int)
            
            self.features_created.extend(['good_for_salads', 'good_for_soups'])
        
        return data
    
    def _create_weather_aggregations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des agrégations météo par période."""
        self.logger.info("   📊 Agrégations météo...")
        
        # Agrégations sur les dernières périodes
        weather_vars = ['temperature', 'precipitation']
        existing_vars = [var for var in weather_vars if var in data.columns]
        
        for var in existing_vars:
            for period in [3, 7, 14]:
                # Somme (pour précipitations)
                if var == 'precipitation':
                    sum_col = f'{var}_sum_{period}d'
                    data[sum_col] = data[var].rolling(window=period, min_periods=1).sum()
                    self.features_created.append(sum_col)
                
                # Moyenne
                mean_col = f'{var}_mean_{period}d'
                data[mean_col] = data[var].rolling(window=period, min_periods=1).mean()
                self.features_created.append(mean_col)
                
                # Min/Max
                min_col = f'{var}_min_{period}d'
                max_col = f'{var}_max_{period}d'
                data[min_col] = data[var].rolling(window=period, min_periods=1).min()
                data[max_col] = data[var].rolling(window=period, min_periods=1).max()
                self.features_created.extend([min_col, max_col])
        
        # Compteurs de jours consécutifs avec certaines conditions
        if 'temperature' in data.columns:
            # Jours consécutifs avec T > 25°C
            data['hot_streak'] = self._calculate_streak(data['temperature'] > 25)
            
            # Jours consécutifs avec T < 5°C
            data['cold_streak'] = self._calculate_streak(data['temperature'] < 5)
            
            self.features_created.extend(['hot_streak', 'cold_streak'])
        
        if 'precipitation' in data.columns:
            # Jours consécutifs sans pluie
            data['dry_streak'] = self._calculate_streak(data['precipitation'] <= 0.1)
            
            # Jours consécutifs avec pluie
            data['wet_streak'] = self._calculate_streak(data['precipitation'] > 0.1)
            
            self.features_created.extend(['dry_streak', 'wet_streak'])
        
        return data
    
    def get_features_list(self) -> List[str]:
        """Retourne la liste des features créées."""
        return self.features_created