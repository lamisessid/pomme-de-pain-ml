"""
Transformer pour les features d'événements.
"""

from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import logging
from datetime import timedelta


class EventsTransformer:
    """Crée des features avancées pour les événements."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.features_created = []
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des features avancées pour les événements.
        
        Features créées:
        - Intensité et propagation d'impact
        - Features temporelles d'événements
        - Interactions entre événements
        - Features spécifiques SQY
        """
        self.logger.info("🎭 Création des features événements avancées...")
        
        if data is None or data.empty or 'has_event' not in data.columns:
            self.logger.warning("⚠️  Données événements vides ou format incorrect")
            # Créer un DataFrame minimal avec des features par défaut
            return self._create_default_event_features(data)
        
        data = data.copy()
        
        # 0. S'assurer que les colonnes de base existent
        data = self._ensure_basic_columns(data)
        
        # 1. Features d'intensité d'événement
        data = self._create_event_intensity_features(data)
        
        # 2. Propagation temporelle d'impact
        data = self._propagate_event_impact(data)
        
        # 3. Features temporelles spécifiques aux événements
        data = self._create_event_temporal_features(data)
        
        # 4. Interactions entre types d'événements
        data = self._create_event_interaction_features(data)
        
        # 5. Features spécifiques à SQY
        data = self._create_sqy_specific_features(data)
        
        # 6. Features d'anticipation et rémanence
        data = self._create_anticipation_features(data)
        
        self.logger.info(f"✅ {len(self.features_created)} features événements créées")
        
        return data
    
    def _ensure_basic_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """S'assure que les colonnes de base existent."""
        # Vérifier et créer 'day_of_week' si nécessaire
        if 'date' in data.columns and 'day_of_week' not in data.columns:
            data['day_of_week'] = data['date'].dt.dayofweek
            self.logger.info("   📅 Colonne 'day_of_week' créée à partir de la date")
        
        # Vérifier et créer 'is_weekend' si nécessaire
        if 'day_of_week' in data.columns and 'is_weekend' not in data.columns:
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            self.logger.info("   🏖️  Colonne 'is_weekend' créée")
        
        # Vérifier et créer 'month' si nécessaire
        if 'date' in data.columns and 'month' not in data.columns:
            data['month'] = data['date'].dt.month
            self.logger.info("   📆 Colonne 'month' créée à partir de la date")
        
        return data
    
    def _create_default_event_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features événements par défaut si pas d'événements."""
        if data is None or 'date' not in data.columns:
            # Créer un DataFrame de base avec dates
            start_date = pd.Timestamp('2022-01-01')
            end_date = pd.Timestamp('2023-12-31')
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            data = pd.DataFrame({'date': date_range})
        
        # S'assurer des colonnes de base
        data = self._ensure_basic_columns(data)
        
        # Features par défaut (toutes à 0 ou 'none')
        default_features = {
            'has_event': 0,
            'event_count': 0,
            'main_event_type': 'none',
            'total_impact': 0.0,
            'max_impact': 0.0,
            'events_list': '',
            'propagated_impact': 0.0,
            'combined_impact': 0.0,
            'impact_category': 'none',
            'impact_ma_7d': 0.0,
            'event_intensity': 0.0,
            'event_complexity': 0.0,
            'days_since_last_event': 999,
            'days_until_next_event': 999,
            'event_cluster_size': 0,
            'has_match': 0,
            'has_university_event': 0,
            'has_market': 0,
            'event_anticipation_effect': 0.0,
            'event_remanence_effect': 0.0,
        }
        
        for feature, default_value in default_features.items():
            data[feature] = default_value
        
        self.features_created = list(default_features.keys())
        
        return data
    
    def _create_event_intensity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features d'intensité d'événement."""
        self.logger.info("   📊 Intensité d'événement...")
        
        # 1. Intensité normalisée (0-1)
        if 'max_impact' in data.columns:
            if data['max_impact'].max() > 0:
                data['event_intensity'] = data['max_impact'] / data['max_impact'].max()
            else:
                data['event_intensity'] = 0
            
            # Catégories d'intensité
            intensity_bins = [-0.1, 0.1, 0.3, 0.6, 1.0]
            intensity_labels = ['none', 'low', 'medium', 'high']
            data['intensity_category'] = pd.cut(
                data['event_intensity'], 
                bins=intensity_bins, 
                labels=intensity_labels
            )
            
            self.features_created.extend(['event_intensity', 'intensity_category'])
        
        # 2. Complexité d'événement (nombre et diversité)
        if 'event_count' in data.columns and 'events_list' in data.columns:
            data['event_complexity'] = data['event_count']
            
            # Si plusieurs événements, augmenter la complexité
            data.loc[data['event_count'] > 1, 'event_complexity'] = (
                data.loc[data['event_count'] > 1, 'event_count'] * 1.5
            )
            
            # Normaliser
            if data['event_complexity'].max() > 0:
                data['event_complexity'] = data['event_complexity'] / data['event_complexity'].max()
            
            self.features_created.append('event_complexity')
        
        # 3. Impact pondéré par type d'événement
        if 'main_event_type' in data.columns and 'max_impact' in data.columns:
            # Poids par type d'événement (basé sur l'impact attendu)
            type_weights = {
                'match': 1.2,           # Fort impact
                'concert': 1.3,         # Très fort impact
                'festival': 1.5,        # Impact majeur
                'demonstration': 0.9,   # Impact modéré mais perturbateur
                'transport_strike': 1.1, # Impact fort sur déplacements
                'conference': 0.8,
                'exhibition': 0.7,
                'university_event': 0.6,
                'market': 0.5,
                'construction': 0.4,
                'other': 0.3
            }
            
            data['weighted_impact'] = data['max_impact'] * data['main_event_type'].map(
                lambda x: type_weights.get(x, 0.5)
            )
            
            self.features_created.append('weighted_impact')
        
        return data
    
    def _propagate_event_impact(self, data: pd.DataFrame) -> pd.DataFrame:
        """Propage l'impact des événements dans le temps."""
        self.logger.info("   🔄 Propagation d'impact...")
        
        # Si pas déjà fait par le cleaner
        if 'propagated_impact' not in data.columns:
            data['propagated_impact'] = data.get('max_impact', 0)
        
        # Propagation avant l'événement (anticipation)
        data['impact_before'] = 0.0
        
        # Propagation après l'événement (rémanence)
        data['impact_after'] = 0.0
        
        # Pour chaque événement, propager l'impact
        event_indices = data[data['has_event'] == 1].index
        
        for idx in event_indices:
            impact = data.at[idx, 'max_impact']
            event_type = data.at[idx, 'main_event_type']
            
            # Définir le rayon de propagation par type d'événement
            propagation_params = self._get_propagation_params(event_type)
            days_before = propagation_params['days_before']
            days_after = propagation_params['days_after']
            decay_rate = propagation_params['decay_rate']
            
            # Propagation avant (anticipation)
            for offset in range(1, days_before + 1):
                if idx - offset >= 0:
                    decay_factor = 1.0 / (offset * decay_rate)
                    current_value = data.at[idx - offset, 'impact_before']
                    new_value = max(current_value, impact * decay_factor)
                    data.at[idx - offset, 'impact_before'] = new_value
            
            # Propagation après (rémanence)
            for offset in range(1, days_after + 1):
                if idx + offset < len(data):
                    decay_factor = 1.0 / (offset * decay_rate)
                    current_value = data.at[idx + offset, 'impact_after']
                    new_value = max(current_value, impact * decay_factor)
                    data.at[idx + offset, 'impact_after'] = new_value
        
        # Impact total propagé
        data['total_propagated_impact'] = (
            data['propagated_impact'] + 
            data['impact_before'] + 
            data['impact_after']
        )
        
        self.features_created.extend([
            'impact_before', 'impact_after', 'total_propagated_impact'
        ])
        
        return data
    
    def _get_propagation_params(self, event_type: str) -> Dict:
        """Retourne les paramètres de propagation par type d'événement."""
        params = {
            'match': {'days_before': 1, 'days_after': 1, 'decay_rate': 1.5},
            'concert': {'days_before': 2, 'days_after': 1, 'decay_rate': 1.2},
            'festival': {'days_before': 3, 'days_after': 2, 'decay_rate': 1.0},
            'demonstration': {'days_before': 1, 'days_after': 1, 'decay_rate': 2.0},
            'transport_strike': {'days_before': 1, 'days_after': 1, 'decay_rate': 1.8},
            'conference': {'days_before': 1, 'days_after': 0, 'decay_rate': 2.0},
            'exhibition': {'days_before': 1, 'days_after': 0, 'decay_rate': 2.0},
            'university_event': {'days_before': 1, 'days_after': 0, 'decay_rate': 2.0},
            'market': {'days_before': 0, 'days_after': 0, 'decay_rate': 1.0},
            'construction': {'days_before': 7, 'days_after': 7, 'decay_rate': 1.5},
            'other': {'days_before': 0, 'days_after': 0, 'decay_rate': 1.0}
        }
        
        return params.get(event_type, params['other'])
    
    def _create_event_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features temporelles pour les événements."""
        self.logger.info("   📅 Features temporelles événements...")
        
        # 1. Temps depuis/avant événement
        data['days_since_last_event'] = 0
        data['days_until_next_event'] = 0
        
        event_indices = data[data['has_event'] == 1].index
        
        if len(event_indices) > 0:
            # Calculer les jours depuis le dernier événement
            last_event_idx = -1
            for i in range(len(data)):
                if i in event_indices:
                    last_event_idx = i
                
                if last_event_idx >= 0:
                    data.at[i, 'days_since_last_event'] = i - last_event_idx
            
            # Calculer les jours jusqu'au prochain événement
            next_event_idx = len(data)
            for i in range(len(data) - 1, -1, -1):
                if i in event_indices:
                    next_event_idx = i
                
                data.at[i, 'days_until_next_event'] = next_event_idx - i
        
        # 2. Fréquence d'événements récente
        for window in [7, 14, 30]:
            col_name = f'event_frequency_{window}d'
            data[col_name] = data['has_event'].rolling(window=window, min_periods=1).sum()
            self.features_created.append(col_name)
        
        # 3. Saisonnalité des événements
        if 'date' in data.columns:
            # Assurer que 'month' existe
            if 'month' not in data.columns:
                data['month'] = data['date'].dt.month
            
            # Densité d'événements par mois (normalisée)
            monthly_event_density = data.groupby('month')['has_event'].transform('mean')
            data['event_monthly_density'] = monthly_event_density
            
            # Anomalie d'événements (écart à la densité mensuelle)
            data['event_anomaly'] = data['has_event'] - monthly_event_density
            
            self.features_created.extend(['event_monthly_density', 'event_anomaly'])
        
        # 4. Patterns hebdomadaires
        if 'day_of_week' in data.columns:
            # Probabilité d'événement par jour de semaine
            dow_event_prob = data.groupby('day_of_week')['has_event'].transform('mean')
            data['dow_event_probability'] = dow_event_prob
            
            # Écart à la probabilité attendue
            data['dow_event_deviation'] = data['has_event'] - dow_event_prob
            
            self.features_created.extend(['dow_event_probability', 'dow_event_deviation'])
        
        self.features_created.extend(['days_since_last_event', 'days_until_next_event'])
        
        return data
    
    def _create_event_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features d'interaction entre événements."""
        self.logger.info("   🔗 Interactions entre événements...")
        
        # 1. Cluster d'événements (événements proches dans le temps)
        cluster_window = 3  # jours
        data['event_cluster_size'] = 0
        data['event_cluster_impact'] = 0.0
        
        for i in range(len(data)):
            # Vérifier les jours autour
            start_idx = max(0, i - cluster_window)
            end_idx = min(len(data), i + cluster_window + 1)
            
            # Compter les événements dans la fenêtre
            cluster_events = data.loc[start_idx:end_idx, 'has_event'].sum()
            cluster_impact = data.loc[start_idx:end_idx, 'max_impact'].max()
            
            data.at[i, 'event_cluster_size'] = cluster_events
            data.at[i, 'event_cluster_impact'] = cluster_impact
        
        # 2. Interactions entre types d'événements
        event_types = ['match', 'concert', 'festival', 'demonstration', 
                      'university_event', 'market', 'transport_strike']
        
        # Indicateurs binaires par type
        for event_type in event_types:
            col_name = f'has_{event_type}'
            data[col_name] = (
                data['main_event_type'] == event_type
            ).astype(int)
            self.features_created.append(col_name)
        
        # 3. Combinaisons d'événements
        if 'events_list' in data.columns:
            # Nombre de types d'événements différents
            def count_unique_types(event_list):
                if not isinstance(event_list, str) or event_list == '':
                    return 0
                # Compter les types uniques (simplifié)
                events = event_list.split(';')
                types = set()
                for event in events:
                    # Extraire le type (simplification)
                    if 'match' in event.lower():
                        types.add('match')
                    elif 'concert' in event.lower():
                        types.add('concert')
                    elif 'festival' in event.lower():
                        types.add('festival')
                return len(types)
            
            data['unique_event_types'] = data['events_list'].apply(count_unique_types)
            self.features_created.append('unique_event_types')
        
        # 4. Séquences d'événements
        if 'has_event' in data.columns:
            # Séquences d'événements consécutifs
            data['event_streak'] = 0
            current_streak = 0
            
            for i in range(len(data)):
                if data.at[i, 'has_event'] == 1:
                    current_streak += 1
                else:
                    current_streak = 0
                data.at[i, 'event_streak'] = current_streak
            
            self.features_created.append('event_streak')
        
        return data
    
    def _create_sqy_specific_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features spécifiques à Saint-Quentin-en-Yvelines."""
        self.logger.info("   🏟️  Features spécifiques SQY...")
        
        # 1. Features pour les matchs (Stade SQY)
        if 'has_match' in data.columns:
            # Impact spécifique match (week-end vs semaine)
            data['match_weekend_impact'] = data['has_match'] * data.get('is_weekend', 0)
            
            # Match en soirée (hypothèse)
            data['match_evening'] = data['has_match']  # À raffiner avec heures réelles
            
            # Match important (hypothèse: week-end = plus important)
            data['match_importance'] = data['has_match'] * (1 + data.get('is_weekend', 0))
            
            self.features_created.extend([
                'match_weekend_impact', 'match_evening', 'match_importance'
            ])
        
        # 2. Features pour événements universitaires (UVSQ)
        if 'has_university_event' in data.columns:
            # Périodes d'examens (déjà dans temporal, mais on ajoute l'impact)
            if 'is_exam_period' in data.columns:
                data['university_exam_impact'] = (
                    data['has_university_event'] * data['is_exam_period']
                )
                self.features_created.append('university_exam_impact')
            
            # Rentrée universitaire (septembre)
            if 'month' in data.columns:
                data['university_start'] = (
                    (data['month'] == 9) & (data['has_university_event'] == 1)
                ).astype(int)
                self.features_created.append('university_start')
        
        # 3. Features pour marchés
        if 'has_market' in data.columns:
            # Impact du marché (plus fort le matin)
            data['market_morning_impact'] = data['has_market']
            
            # Marché + beau temps = plus d'impact
            if 'is_sunny_day' in data.columns:
                data['market_sunny_impact'] = (
                    data['has_market'] * data['is_sunny_day']
                )
                self.features_created.append('market_sunny_impact')
        
        # 4. Interactions événements × météo
        if 'has_event' in data.columns:
            # Événement + mauvais temps = impact différent
            if 'is_rainy_day' in data.columns:
                data['event_rain_impact'] = (
                    data['has_event'] * data['is_rainy_day']
                )
                self.features_created.append('event_rain_impact')
            
            # Événement + beau temps = terrasse
            if 'good_for_dine_in' in data.columns:
                data['event_good_weather_impact'] = (
                    data['has_event'] * data['good_for_dine_in']
                )
                self.features_created.append('event_good_weather_impact')
        
        # 5. Double clientèle SQY (bureau + événementiel)
        # CORRECTION : Vérifier que 'day_of_week' et 'is_weekend' existent
        if 'day_of_week' in data.columns:
            data['double_audience_day'] = (
                (data['has_event'] == 1) & 
                (data['day_of_week'] < 5)  # Jour de semaine
            ).astype(int)
        else:
            data['double_audience_day'] = 0
        
        if 'is_weekend' in data.columns:
            data['weekend_event_day'] = (
                (data['has_event'] == 1) & 
                (data['is_weekend'] == 1)
            ).astype(int)
        else:
            data['weekend_event_day'] = 0
        
        self.features_created.extend(['double_audience_day', 'weekend_event_day'])
        
        return data
    
    def _create_anticipation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features d'anticipation et rémanence."""
        self.logger.info("   ⏳ Anticipation et rémanence...")
        
        # 1. Effet d'anticipation (préparation avant événement)
        data['event_anticipation_effect'] = 0.0
        
        # Pour chaque événement, créer un effet d'anticipation croissant
        event_indices = data[data['has_event'] == 1].index
        
        for idx in event_indices:
            event_type = data.at[idx, 'main_event_type']
            impact = data.at[idx, 'max_impact']
            
            # Paramètres d'anticipation par type
            anticipation_days = self._get_anticipation_days(event_type)
            
            for offset in range(1, anticipation_days + 1):
                if idx - offset >= 0:
                    # Effet croissant à mesure qu'on approche de l'événement
                    anticipation_effect = impact * (offset / anticipation_days)
                    
                    # Ajouter à l'effet existant
                    current = data.at[idx - offset, 'event_anticipation_effect']
                    data.at[idx - offset, 'event_anticipation_effect'] = max(
                        current, anticipation_effect
                    )
        
        # 2. Effet de rémanence (après événement)
        data['event_remanence_effect'] = 0.0
        
        for idx in event_indices:
            event_type = data.at[idx, 'main_event_type']
            impact = data.at[idx, 'max_impact']
            
            # Paramètres de rémanence
            remanence_days = self._get_remanence_days(event_type)
            
            for offset in range(1, remanence_days + 1):
                if idx + offset < len(data):
                    # Effet décroissant après l'événement
                    remanence_effect = impact * (1 - (offset / (remanence_days + 1)))
                    
                    current = data.at[idx + offset, 'event_remanence_effect']
                    data.at[idx + offset, 'event_remanence_effect'] = max(
                        current, remanence_effect
                    )
        
        # 3. Indicateur de période "autour d'événement"
        data['around_event_period'] = (
            (data['event_anticipation_effect'] > 0) |
            (data['event_remanence_effect'] > 0) |
            (data['has_event'] == 1)
        ).astype(int)
        
        # 4. Impact total avec anticipation et rémanence
        if 'total_propagated_impact' in data.columns:
            data['total_event_effect'] = (
                data['total_propagated_impact'] +
                data['event_anticipation_effect'] +
                data['event_remanence_effect']
            )
            self.features_created.append('total_event_effect')
        
        self.features_created.extend([
            'event_anticipation_effect', 
            'event_remanence_effect',
            'around_event_period'
        ])
        
        return data
    
    def _get_anticipation_days(self, event_type: str) -> int:
        """Retourne le nombre de jours d'anticipation par type d'événement."""
        anticipation_map = {
            'festival': 7,
            'concert': 3,
            'match': 2,
            'demonstration': 1,
            'transport_strike': 1,
            'conference': 2,
            'exhibition': 2,
            'university_event': 1,
            'market': 0,
            'construction': 0,
            'other': 0
        }
        return anticipation_map.get(event_type, 0)
    
    def _get_remanence_days(self, event_type: str) -> int:
        """Retourne le nombre de jours de rémanence par type d'événement."""
        remanence_map = {
            'festival': 3,
            'concert': 2,
            'match': 1,
            'demonstration': 1,
            'transport_strike': 1,
            'conference': 1,
            'exhibition': 1,
            'university_event': 1,
            'market': 0,
            'construction': 7,  # Travaux durent
            'other': 0
        }
        return remanence_map.get(event_type, 0)
    
    def get_features_list(self) -> List[str]:
        """Retourne la liste des features créées."""
        return self.features_created