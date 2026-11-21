"""
Transformer pour les features temporelles.
"""

from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import logging

from ..utils import create_calendar_features
from ..config import processing_config


class TemporalTransformer:
    """Crée des features temporelles."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.features_created = []
    
    def transform(self, data: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Ajoute des features temporelles aux données.
        
        Features créées:
        - Features calendrier basiques
        - Features cycliques (sin/cos)
        - Jours fériés français
        - Vacances scolaires (Zone C)
        - Features de lag
        - Features de moyenne mobile
        """
        self.logger.info("📅 Création des features temporelles...")
        
        if data is None or data.empty:
            self.logger.warning("⚠️  Données vides pour transformation temporelle")
            return data
        
        if date_column not in data.columns:
            self.logger.error(f"❌ Colonne {date_column} non trouvée")
            raise ValueError(f"Colonne {date_column} non trouvée")
        
        data = data.copy()
        
        # 1. S'assurer que la date est au format datetime
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        
        # 2. Features calendrier basiques
        data = create_calendar_features(data, date_column)
        self.features_created.extend([
            'year', 'month', 'day', 'day_of_week', 'day_of_year',
            'week_of_year', 'quarter', 'is_weekend', 'is_month_start',
            'is_month_end', 'is_quarter_start', 'is_quarter_end',
            'is_year_start', 'is_year_end'
        ])
        
        # 3. Features cycliques
        data = self._create_cyclic_features(data)
        
        # 4. Jours fériés français
        data = self._add_french_holidays(data, date_column)
        
        # 5. Vacances scolaires Zone C (Versailles)
        data = self._add_school_holidays(data, date_column)
        
        # 6. Features de lag (si données de ventes)
        if 'quantity' in data.columns:
            data = self._create_lag_features(data, 'quantity')
        
        # 7. Features de moyenne mobile
        if 'quantity' in data.columns:
            data = self._create_rolling_features(data, 'quantity')
        
        # 8. Features de saisonnalité
        data = self._create_seasonality_features(data, date_column)
        
        # 9. Distance aux événements spéciaux
        data = self._add_special_dates(data, date_column)
        
        self.logger.info(f"✅ {len(self.features_created)} features temporelles créées")
        
        return data
    
    def _create_cyclic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features cycliques pour les modèles ML."""
        # Mois (déjà créé par create_calendar_features mais on ajoute plus)
        if 'month' in data.columns:
            data['month_sin_12'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos_12'] = np.cos(2 * np.pi * data['month'] / 12)
            data['month_sin_6'] = np.sin(2 * np.pi * data['month'] / 6)
            data['month_cos_6'] = np.cos(2 * np.pi * data['month'] / 6)
            self.features_created.extend(['month_sin_12', 'month_cos_12', 'month_sin_6', 'month_cos_6'])
        
        # Jour de la semaine
        if 'day_of_week' in data.columns:
            data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            self.features_created.extend(['day_of_week_sin', 'day_of_week_cos'])
        
        # Jour de l'année
        if 'day_of_year' in data.columns:
            data['day_of_year_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
            data['day_of_year_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
            self.features_created.extend(['day_of_year_sin', 'day_of_year_cos'])
        
        # Heure du jour (si données horaires)
        if 'hour' in data.columns:
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            self.features_created.extend(['hour_sin', 'hour_cos'])
        
        return data
    
    def _add_french_holidays(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Ajoute les jours fériés français."""
        from datetime import datetime
        
        self.logger.info("   🎉 Ajout des jours fériés français...")
        
        # Initialiser la colonne
        data['is_holiday'] = 0
        data['holiday_name'] = ''
        
        # Extraire l'année des dates
        years = data[date_column].dt.year.unique()
        
        # Jours fériés fixes
        fixed_holidays = {
            '01-01': "Jour de l'an",
            '05-01': "Fête du Travail",
            '05-08': "Victoire 1945",
            '07-14': "Fête Nationale",
            '08-15': "Assomption",
            '11-01': "Toussaint",
            '11-11': "Armistice 1918",
            '12-25': "Noël"
        }
        
        # Jours fériés mobiles (à calculer pour chaque année)
        def get_easter_date(year):
            """Calcule la date de Pâques (algorithme de Gauss)."""
            a = year % 19
            b = year // 100
            c = year % 100
            d = b // 4
            e = b % 4
            f = (b + 8) // 25
            g = (b - f + 1) // 3
            h = (19 * a + b - d - g + 15) % 30
            i = c // 4
            k = c % 4
            l = (32 + 2 * e + 2 * i - h - k) % 7
            m = (a + 11 * h + 22 * l) // 451
            month = (h + l - 7 * m + 114) // 31
            day = ((h + l - 7 * m + 114) % 31) + 1
            return datetime(year, month, day)
        
        # Pour chaque année dans les données
        for year in years:
            easter = get_easter_date(year)
            
            # Jours fériés mobiles relatifs à Pâques
            mobile_holidays = {
                easter: "Pâques",
                easter + pd.Timedelta(days=1): "Lundi de Pâques",
                easter + pd.Timedelta(days=39): "Ascension",
                easter + pd.Timedelta(days=49): "Pentecôte",
                easter + pd.Timedelta(days=50): "Lundi de Pentecôte"
            }
            
            # Marquer les jours fériés fixes
            for date_str, name in fixed_holidays.items():
                holiday_date = datetime.strptime(f"{year}-{date_str}", "%Y-%m-%d")
                mask = data[date_column].dt.date == holiday_date.date()
                data.loc[mask, 'is_holiday'] = 1
                data.loc[mask, 'holiday_name'] = name
            
            # Marquer les jours fériés mobiles
            for holiday_date, name in mobile_holidays.items():
                mask = data[date_column].dt.date == holiday_date.date()
                data.loc[mask, 'is_holiday'] = 1
                data.loc[mask, 'holiday_name'] = name
        
        # Indicateur pour la veille d'un jour férié
        data['is_before_holiday'] = data['is_holiday'].shift(-1).fillna(0).astype(int)
        
        # Indicateur pour le lendemain d'un jour férié
        data['is_after_holiday'] = data['is_holiday'].shift(1).fillna(0).astype(int)
        
        self.features_created.extend(['is_holiday', 'is_before_holiday', 'is_after_holiday'])
        
        holiday_count = data['is_holiday'].sum()
        self.logger.info(f"     - {holiday_count} jours fériés identifiés")
        
        return data
    
    def _add_school_holidays(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Ajoute les vacances scolaires (Zone C - Versailles)."""
        self.logger.info("   🏫 Ajout des vacances scolaires Zone C...")
        
        # Périodes de vacances scolaires pour Zone C (approximatives, à adapter)
        # Format: (début, fin, nom)
        school_holidays = [
            # 2022
            ("2022-02-12", "2022-02-28", "Vacances d'hiver"),
            ("2022-04-16", "2022-05-02", "Vacances de printemps"),
            ("2022-07-07", "2022-09-01", "Vacances d'été"),
            ("2022-10-22", "2022-11-07", "Vacances de la Toussaint"),
            ("2022-12-17", "2023-01-03", "Vacances de Noël"),
            
            # 2023
            ("2023-02-11", "2023-02-27", "Vacances d'hiver"),
            ("2023-04-08", "2023-04-24", "Vacances de printemps"),
            ("2023-07-08", "2023-09-04", "Vacances d'été"),
            ("2023-10-21", "2023-11-06", "Vacances de la Toussaint"),
            ("2023-12-23", "2024-01-08", "Vacances de Noël"),
        ]
        
        # Initialiser les colonnes
        data['is_school_holiday'] = 0
        data['school_holiday_name'] = ''
        data['days_into_holiday'] = 0
        data['days_until_holiday'] = 0
        
        # Marquer les périodes de vacances
        for start_str, end_str, name in school_holidays:
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
            
            mask = (data[date_column] >= start_date) & (data[date_column] <= end_date)
            data.loc[mask, 'is_school_holiday'] = 1
            data.loc[mask, 'school_holiday_name'] = name
            
            # Calculer les jours dans les vacances
            data.loc[mask, 'days_into_holiday'] = (data.loc[mask, date_column] - start_date).dt.days + 1
        
        # Calculer les jours avant les prochaines vacances
        for idx, row in data.iterrows():
            current_date = row[date_column]
            
            # Trouver la prochaine période de vacances
            next_holiday = None
            for start_str, end_str, name in school_holidays:
                start_date = pd.to_datetime(start_str)
                if start_date > current_date:
                    next_holiday = start_date
                    break
            
            if next_holiday is not None:
                data.at[idx, 'days_until_holiday'] = (next_holiday - current_date).days
        
        # Indicateur pour la rentrée (premier jour après vacances)
        data['is_first_day_after_holiday'] = (
            (data['is_school_holiday'] == 0) & 
            (data['is_school_holiday'].shift(1) == 1)
        ).astype(int)
        
        # Indicateur pour la veille des vacances
        data['is_day_before_holiday'] = (
            (data['is_school_holiday'] == 0) & 
            (data['is_school_holiday'].shift(-1) == 1)
        ).astype(int)
        
        self.features_created.extend([
            'is_school_holiday', 'days_into_holiday', 'days_until_holiday',
            'is_first_day_after_holiday', 'is_day_before_holiday'
        ])
        
        holiday_days = data['is_school_holiday'].sum()
        self.logger.info(f"     - {holiday_days} jours de vacances scolaires")
        
        return data
    
    def _create_lag_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Crée des features de lag."""
        self.logger.info(f"   ⏪ Création des lag features pour {target_column}...")
        
        # Lags standards
        lags = processing_config.lag_features
        
        for lag in lags:
            col_name = f'{target_column}_lag_{lag}'
            data[col_name] = data[target_column].shift(lag)
            self.features_created.append(col_name)
        
        # Lags différenciés (changements)
        for lag in [1, 7]:
            col_name = f'{target_column}_diff_{lag}'
            data[col_name] = data[target_column].diff(lag)
            self.features_created.append(col_name)
        
        # Ratio par rapport à la même période
        for period in [7, 30]:
            col_name = f'{target_column}_ratio_{period}d'
            data[col_name] = data[target_column] / data[target_column].shift(period)
            data[col_name] = data[col_name].replace([np.inf, -np.inf], np.nan)
            self.features_created.append(col_name)
        
        self.logger.info(f"     - {len(lags) + 4} lag features créées")
        
        return data
    
    def _create_rolling_features(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Crée des features de moyenne mobile."""
        self.logger.info(f"   📊 Création des rolling features pour {target_column}...")
        
        rolling_windows = processing_config.rolling_windows
        
        for window in rolling_windows:
            # Moyenne mobile
            mean_col = f'{target_column}_ma_{window}d'
            data[mean_col] = data[target_column].rolling(window=window, min_periods=1).mean()
            self.features_created.append(mean_col)
            
            # Écart-type mobile
            std_col = f'{target_column}_std_{window}d'
            data[std_col] = data[target_column].rolling(window=window, min_periods=1).std()
            self.features_created.append(std_col)
            
            # Min/max mobile
            min_col = f'{target_column}_min_{window}d'
            max_col = f'{target_column}_max_{window}d'
            data[min_col] = data[target_column].rolling(window=window, min_periods=1).min()
            data[max_col] = data[target_column].rolling(window=window, min_periods=1).max()
            self.features_created.extend([min_col, max_col])
            
            # Percentile mobile
            for percentile in [25, 50, 75]:
                pct_col = f'{target_column}_p{percentile}_{window}d'
                data[pct_col] = data[target_column].rolling(window=window, min_periods=1).quantile(percentile/100)
                self.features_created.append(pct_col)
        
        # Tendances (pente sur fenêtre glissante)
        for window in [7, 14]:
            trend_col = f'{target_column}_trend_{window}d'
            data[trend_col] = data[target_column].rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan,
                raw=True
            )
            self.features_created.append(trend_col)
        
        self.logger.info(f"     - {len(rolling_windows) * 7 + 2} rolling features créées")
        
        return data
    
    def _create_seasonality_features(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Crée des features de saisonnalité."""
        self.logger.info("   🌸 Création des features de saisonnalité...")
        
        # Saison (déjà créée mais on ajoute des indicateurs)
        seasons = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        }
        
        data['season'] = data[date_column].dt.month.map(seasons)
        
        # Indicateurs de saison
        for season in ['winter', 'spring', 'summer', 'autumn']:
            col_name = f'is_{season}'
            data[col_name] = (data['season'] == season).astype(int)
            self.features_created.append(col_name)
        
        # Mois de l'année (one-hot encoding optionnel)
        # data = pd.get_dummies(data, columns=['month'], prefix='month')
        
        # Trimestre (one-hot encoding optionnel)
        # data = pd.get_dummies(data, columns=['quarter'], prefix='q')
        
        # Jour de la semaine (one-hot encoding optionnel)
        # data = pd.get_dummies(data, columns=['day_of_week'], prefix='dow')
        
        # Week-end étendu (vendredi à dimanche)
        data['is_extended_weekend'] = data['day_of_week'].isin([4, 5, 6]).astype(int)
        self.features_created.append('is_extended_weekend')
        
        # Jours de paie (approximatifs)
        data['is_payday'] = data['day'].isin([1, 15]).astype(int)
        self.features_created.append('is_payday')
        
        # Période de fin de mois
        data['is_end_of_month'] = (data['day'] >= 25).astype(int)
        self.features_created.append('is_end_of_month')
        
        self.logger.info(f"     - 7 features de saisonnalité créées")
        
        return data
    
    def _add_special_dates(self, data: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Ajoute des dates spéciales spécifiques à SQY."""
        self.logger.info("   🎪 Ajout des dates spéciales SQY...")
        
        # Jours de marché à Saint-Quentin (ex: mercredi et samedi)
        data['is_market_day'] = data['day_of_week'].isin([2, 5]).astype(int)  # Mercredi=2, Samedi=5
        self.features_created.append('is_market_day')
        
        # Périodes d'examens UVSQ (approximatives)
        exam_periods = [
            ("2022-01-10", "2022-01-28", "Examens session 1"),
            ("2022-05-15", "2022-06-10", "Examens session 2"),
            ("2022-08-20", "2022-09-10", "Rattrapages"),
            ("2023-01-09", "2023-01-27", "Examens session 1"),
            ("2023-05-15", "2023-06-09", "Examens session 2"),
            ("2023-08-21", "2023-09-11", "Rattrapages"),
        ]
        
        data['is_exam_period'] = 0
        data['exam_period_name'] = ''
        
        for start_str, end_str, name in exam_periods:
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
            
            mask = (data[date_column] >= start_date) & (data[date_column] <= end_date)
            data.loc[mask, 'is_exam_period'] = 1
            data.loc[mask, 'exam_period_name'] = name
        
        # Matchs au stade (dates approximatives)
        match_dates = [
            "2022-01-15", "2022-01-29", "2022-02-12", "2022-03-05",
            "2022-03-19", "2022-04-02", "2022-04-16", "2022-04-30",
            "2022-05-14", "2022-05-28", "2022-06-11", "2022-09-03",
            "2022-09-17", "2022-10-01", "2022-10-15", "2022-10-29",
            "2022-11-12", "2022-11-26", "2022-12-10", "2023-01-14",
            "2023-01-28", "2023-02-11", "2023-03-04", "2023-03-18",
            "2023-04-01", "2023-04-15", "2023-04-29", "2023-05-13",
            "2023-05-27", "2023-06-10"
        ]
        
        data['is_match_day'] = 0
        for date_str in match_dates:
            match_date = pd.to_datetime(date_str)
            mask = data[date_column].dt.date == match_date.date()
            data.loc[mask, 'is_match_day'] = 1
        
        # Jours avant/après match
        data['days_since_last_match'] = 0
        data['days_until_next_match'] = 0
        
        match_dates_series = pd.to_datetime(match_dates)
        
        for idx, row in data.iterrows():
            current_date = row[date_column]
            
            # Jours depuis le dernier match
            past_matches = match_dates_series[match_dates_series < current_date]
            if not past_matches.empty:
                last_match = past_matches.max()
                data.at[idx, 'days_since_last_match'] = (current_date - last_match).days
            
            # Jours jusqu'au prochain match
            future_matches = match_dates_series[match_dates_series > current_date]
            if not future_matches.empty:
                next_match = future_matches.min()
                data.at[idx, 'days_until_next_match'] = (next_match - current_date).days
        
        self.features_created.extend([
            'is_market_day', 'is_exam_period', 'is_match_day',
            'days_since_last_match', 'days_until_next_match'
        ])
        
        self.logger.info("     - Dates spéciales SQY ajoutées")
        
        return data
    
    def get_features_list(self) -> List[str]:
        """Retourne la liste des features créées."""
        return self.features_created