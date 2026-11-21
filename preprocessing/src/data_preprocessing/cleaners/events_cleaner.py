"""
Cleaner pour les données d'événements.
"""

from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import timedelta

from .base_cleaner import BaseCleaner
from ..config import processing_config


class EventsCleaner(BaseCleaner):
    """Nettoie les données d'événements."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.stats = {
            'events_processed': 0,
            'events_merged': 0,
            'impact_calculated': 0,
            'days_affected': 0
        }
        
        # Mapping d'impact par type d'événement (à adapter)
        self.impact_mapping = {
            'match': {'attendance_weight': 0.8, 'radius_days': 1},
            'concert': {'attendance_weight': 0.7, 'radius_days': 2},
            'festival': {'attendance_weight': 0.9, 'radius_days': 3},
            'conference': {'attendance_weight': 0.5, 'radius_days': 1},
            'exhibition': {'attendance_weight': 0.6, 'radius_days': 1},
            'university_event': {'attendance_weight': 0.4, 'radius_days': 1},
            'market': {'attendance_weight': 0.3, 'radius_days': 1},
            'demonstration': {'attendance_weight': 0.5, 'radius_days': 1},
            'construction': {'attendance_weight': 0.4, 'radius_days': 7},  # Travaux durent longtemps
            'transport_strike': {'attendance_weight': 0.6, 'radius_days': 1},
            'other': {'attendance_weight': 0.3, 'radius_days': 1}
        }
    
    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données d'événements.
        
        Étapes:
        1. Valider la structure
        2. Standardiser les types d'événements
        3. Calculer l'impact estimé
        4. Gérer les événements multiples le même jour
        5. Créer des features d'impact temporel
        """
        self.logger.info("🎭 Nettoyage des données d'événements...")
        
        if data is None or data.empty:
            self.logger.info("ℹ️  Aucun événement à nettoyer, création d'un DataFrame vide")
            return self._create_empty_events_df()
        
        data = data.copy()
        initial_events = len(data)
        
        # 1. Valider la structure
        data = self._validate_structure(data)
        
        # 2. Standardiser les dates
        data = self._standardize_dates(data)
        
        # 3. Standardiser les types d'événements
        data = self._standardize_event_types(data)
        
        # 4. Calculer/standardiser l'impact
        data = self._calculate_impact(data)
        
        # 5. Gérer les événements multiples
        data = self._handle_multiple_events(data)
        
        # 6. Créer des indicateurs temporels
        data = self._create_temporal_features(data)
        
        # 7. Filtrer sur la période configurée
        data = self._filter_by_date_range(data)
        
        # 8. Créer un calendrier d'événements quotidien
        daily_events = self._create_daily_events_calendar(data)
        
        # Statistiques finales
        self.stats['events_processed'] = len(data)
        self.stats['days_affected'] = len(daily_events)
        
        self.logger.info(f"✅ Nettoyage événements terminé: {initial_events} événements")
        self._log_cleaning_stats()
        
        return daily_events
    
    def _create_empty_events_df(self) -> pd.DataFrame:
        """Crée un DataFrame vide pour les événements."""
        columns = [
            'date',
            'has_event',
            'event_count',
            'main_event_type',
            'total_impact',
            'max_impact',
            'events_list'
        ]
        
        # Créer un calendrier vide pour la période
        start_date = pd.to_datetime(processing_config.start_date)
        end_date = pd.to_datetime(processing_config.end_date)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        empty_df = pd.DataFrame({
            'date': date_range,
            'has_event': 0,
            'event_count': 0,
            'main_event_type': 'none',
            'total_impact': 0.0,
            'max_impact': 0.0,
            'events_list': ''
        })
        
        return empty_df
    
    def _validate_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Valide la structure des données d'événements."""
        # Colonnes minimales requises
        if 'date' not in data.columns:
            self.logger.error("❌ Colonne 'date' manquante dans les événements")
            raise ValueError("Colonne 'date' manquante")
        
        # Créer event_name si manquant
        if 'event_name' not in data.columns:
            self.logger.warning("⚠️  Colonne 'event_name' manquante, création")
            data['event_name'] = [f"event_{i}" for i in range(len(data))]
        
        # Créer event_type si manquant
        if 'event_type' not in data.columns:
            self.logger.warning("⚠️  Colonne 'event_type' manquante, mise à 'other'")
            data['event_type'] = 'other'
        
        # Nettoyer les types d'événements
        data['event_type'] = data['event_type'].astype(str).str.lower().str.strip()
        
        # Remplacer les valeurs vides
        data['event_type'] = data['event_type'].replace(['nan', 'none', ''], 'other')
        
        return data
    
    def _standardize_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardise les dates des événements."""
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        
        # Supprimer les événements sans date valide
        invalid_dates = data['date'].isnull().sum()
        if invalid_dates > 0:
            self.logger.warning(f"Suppression de {invalid_dates} événements sans date valide")
            data = data.dropna(subset=['date'])
        
        # S'assurer que la date est au format date seulement (pas heure)
        data['date'] = data['date'].dt.normalize()
        
        return data
    
    def _standardize_event_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardise les types d'événements."""
        # Mapping de standardisation
        type_mapping = {
            'sport': 'match',
            'football': 'match',
            'rugby': 'match',
            'basketball': 'match',
            'tennis': 'match',
            'musique': 'concert',
            'spectacle': 'concert',
            'show': 'concert',
            'fête': 'festival',
            'carnaval': 'festival',
            'conférence': 'conference',
            'colloque': 'conference',
            'séminaire': 'conference',
            'exposition': 'exhibition',
            'salon': 'exhibition',
            'foire': 'exhibition',
            'université': 'university_event',
            'campus': 'university_event',
            'étudiant': 'university_event',
            'exam': 'university_event',
            'marché': 'market',
            'brocante': 'market',
            'vide-grenier': 'market',
            'manifestation': 'demonstration',
            'protestation': 'demonstration',
            'grève': 'transport_strike',
            'sncf': 'transport_strike',
            'rer': 'transport_strike',
            'transport': 'transport_strike',
            'travaux': 'construction',
            'chantier': 'construction',
            'fermeture': 'construction',
        }
        
        # Appliquer le mapping
        data['event_type'] = data['event_type'].map(
            lambda x: type_mapping.get(x, x) if isinstance(x, str) else 'other'
        )
        
        # S'assurer que tous les types sont dans notre mapping d'impact
        unknown_types = set(data['event_type'].unique()) - set(self.impact_mapping.keys())
        if unknown_types:
            self.logger.warning(f"⚠️  Types d'événements inconnus: {unknown_types}")
            # Remplacer par 'other'
            data['event_type'] = data['event_type'].apply(
                lambda x: x if x in self.impact_mapping else 'other'
            )
        
        # Comptage des types
        type_counts = data['event_type'].value_counts()
        self.logger.info(f"   - Types d'événements: {type_counts.to_dict()}")
        
        return data
    
    def _calculate_impact(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calcule l'impact estimé des événements."""
        self.logger.info("   📈 Calcul des impacts d'événements...")
        
        # 1. Standardiser l'affluence estimée
        if 'estimated_attendance' in data.columns:
            # Convertir en numérique
            data['estimated_attendance'] = pd.to_numeric(data['estimated_attendance'], errors='coerce')
            
            # Remplacer les valeurs manquantes par la médiane par type
            for event_type in data['event_type'].unique():
                mask = data['event_type'] == event_type
                median_val = data.loc[mask, 'estimated_attendance'].median()
                if pd.notna(median_val):
                    data.loc[mask & data['estimated_attendance'].isna(), 'estimated_attendance'] = median_val
            
            # Remplacer les valeurs restantes par 0
            data['estimated_attendance'] = data['estimated_attendance'].fillna(0)
            
            # Normaliser l'affluence (0-1)
            if data['estimated_attendance'].max() > 0:
                data['attendance_norm'] = (
                    data['estimated_attendance'] / data['estimated_attendance'].max()
                )
            else:
                data['attendance_norm'] = 0
        else:
            # Pas d'affluence, utiliser des valeurs par défaut par type
            data['attendance_norm'] = data['event_type'].map(
                lambda x: self.impact_mapping.get(x, self.impact_mapping['other'])['attendance_weight']
            )
        
        # 2. Standardiser le niveau d'impact si présent
        if 'impact_level' in data.columns:
            # Mapping des niveaux d'impact
            impact_level_mapping = {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.9,
                'very_high': 1.0,
                'unknown': 0.5
            }
            
            data['impact_level'] = (
                data['impact_level']
                .astype(str)
                .str.lower()
                .str.strip()
                .map(impact_level_mapping)
                .fillna(0.5)
            )
        else:
            # Utiliser l'affluence normalisée comme proxy
            data['impact_level'] = data['attendance_norm']
        
        # 3. Calculer l'impact final (combinaison type + affluence + niveau)
        def calculate_final_impact(row):
            event_type = row['event_type']
            base_params = self.impact_mapping.get(event_type, self.impact_mapping['other'])
            
            # Impact = poids du type × niveau d'impact × affluence
            impact = (
                base_params['attendance_weight'] *
                row['impact_level'] *
                (row.get('attendance_norm', 0.5) if pd.notna(row.get('attendance_norm')) else 0.5)
            )
            
            return min(1.0, max(0.0, impact))  # Borne entre 0 et 1
        
        data['event_impact'] = data.apply(calculate_final_impact, axis=1)
        
        # 4. Calculer le rayon d'impact (en jours)
        data['impact_radius'] = data['event_type'].map(
            lambda x: self.impact_mapping.get(x, self.impact_mapping['other'])['radius_days']
        )
        
        # Statistiques d'impact
        avg_impact = data['event_impact'].mean()
        self.logger.info(f"     - Impact moyen: {avg_impact:.2f}")
        self.logger.info(f"     - Distribution: {data['event_impact'].describe().to_dict()}")
        
        self.stats['impact_calculated'] = len(data)
        
        return data
    
    def _handle_multiple_events(self, data: pd.DataFrame) -> pd.DataFrame:
        """Gère les événements multiples le même jour."""
        # Compter les événements par date avant fusion
        events_per_day = data.groupby('date').size()
        multi_event_days = (events_per_day > 1).sum()
        
        if multi_event_days > 0:
            self.logger.info(f"   - {multi_event_days} jours avec événements multiples")
            
            # Agrégation par date
            aggregated = data.groupby('date').agg({
                'event_name': lambda x: '; '.join(x.astype(str)),
                'event_type': lambda x: self._get_main_event_type(x),
                'event_impact': ['sum', 'max'],
                'impact_radius': 'max'
            })
            
            # Aplatir les colonnes multi-index
            aggregated.columns = [
                'events_list',
                'main_event_type',
                'total_impact',
                'max_impact',
                'impact_radius'
            ]
            
            aggregated = aggregated.reset_index()
            
            # Garder aussi les événements individuels pour référence
            individual_events = data[['date', 'event_name', 'event_type', 'event_impact']].copy()
            
            self.stats['events_merged'] = multi_event_days
            
            return aggregated
        else:
            # Pas d'événements multiples, formater de la même manière
            data['events_list'] = data['event_name']
            data['total_impact'] = data['event_impact']
            data['max_impact'] = data['event_impact']
            data['main_event_type'] = data['event_type']
            
            return data[['date', 'events_list', 'main_event_type', 'total_impact', 'max_impact', 'impact_radius']]
    
    def _get_main_event_type(self, event_types: pd.Series) -> str:
        """Détermine le type d'événement principal parmi plusieurs."""
        # Priorité des types d'événements (impact décroissant)
        type_priority = {
            'festival': 10,
            'concert': 9,
            'match': 8,
            'demonstration': 7,
            'transport_strike': 6,
            'exhibition': 5,
            'conference': 4,
            'construction': 3,
            'university_event': 2,
            'market': 1,
            'other': 0
        }
        
        # Trouver le type avec la priorité la plus haute
        priorities = event_types.map(lambda x: type_priority.get(x, 0))
        return event_types.iloc[priorities.argmax()]
    
    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée des features temporelles pour les événements."""
        from ..utils import create_calendar_features
        
        # Features de calendrier basiques
        data = create_calendar_features(data, 'date')
        
        # Jours depuis dernier événement du même type
        for event_type in data['main_event_type'].unique():
            type_mask = data['main_event_type'] == event_type
            data.loc[type_mask, f'days_since_last_{event_type}'] = (
                data.loc[type_mask, 'date'].diff().dt.days
            )
        
        # Indicateur de weekend avec événement
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        data['weekend_event'] = (data['is_weekend'] == 1).astype(int)
        
        # Saison de l'événement
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
        
        data['event_season'] = data['date'].apply(get_season)
        
        return data
    
    def _create_daily_events_calendar(self, data: pd.DataFrame) -> pd.DataFrame:
        """Crée un calendrier quotidien d'événements avec propagation d'impact."""
        if data.empty:
            return self._create_empty_events_df()
        
        # Créer une série de dates complète
        start_date = pd.to_datetime(processing_config.start_date)
        end_date = pd.to_datetime(processing_config.end_date)
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # DataFrame de base pour tous les jours
        calendar_df = pd.DataFrame({'date': all_dates})
        
        # Initialiser les colonnes
        calendar_df['has_event'] = 0
        calendar_df['event_count'] = 0
        calendar_df['main_event_type'] = 'none'
        calendar_df['total_impact'] = 0.0
        calendar_df['max_impact'] = 0.0
        calendar_df['events_list'] = ''
        
        # Remplir avec les événements exacts
        for _, event in data.iterrows():
            event_date = event['date']
            
            # Trouver l'index dans le calendrier
            if event_date in calendar_df['date'].values:
                idx = calendar_df[calendar_df['date'] == event_date].index[0]
                
                # Mettre à jour les valeurs
                calendar_df.at[idx, 'has_event'] = 1
                calendar_df.at[idx, 'event_count'] += 1
                
                # Mettre à jour le type principal (garder le plus impactant)
                if event['max_impact'] > calendar_df.at[idx, 'max_impact']:
                    calendar_df.at[idx, 'main_event_type'] = event['main_event_type']
                
                # Ajouter l'impact
                calendar_df.at[idx, 'total_impact'] += event['total_impact']
                calendar_df.at[idx, 'max_impact'] = max(
                    calendar_df.at[idx, 'max_impact'], event['max_impact']
                )
                
                # Ajouter à la liste des événements
                if calendar_df.at[idx, 'events_list']:
                    calendar_df.at[idx, 'events_list'] += f"; {event['events_list']}"
                else:
                    calendar_df.at[idx, 'events_list'] = event['events_list']
        
        # Propager l'impact sur les jours autour des événements
        calendar_df = self._propagate_event_impact(calendar_df, data)
        
        # Features dérivées
        calendar_df['impact_category'] = pd.cut(
            calendar_df['max_impact'],
            bins=[-0.1, 0.1, 0.3, 0.6, 1.0],
            labels=['none', 'low', 'medium', 'high']
        )
        
        # Moyenne mobile de l'impact
        calendar_df['impact_ma_7d'] = calendar_df['max_impact'].rolling(7, min_periods=1).mean()
        
        return calendar_df
    
    def _propagate_event_impact(self, calendar_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """Propage l'impact des événements sur les jours autour."""
        self.logger.info("   🔄 Propagation de l'impact des événements...")
        
        # Pour chaque événement, propager l'impact avant/après
        propagated_impact = calendar_df['max_impact'].copy()
        
        for _, event in events_df.iterrows():
            event_date = event['date']
            impact = event['max_impact']
            radius = event.get('impact_radius', 1)
            
            # Trouver l'index de la date de l'événement
            if event_date not in calendar_df['date'].values:
                continue
            
            event_idx = calendar_df[calendar_df['date'] == event_date].index[0]
            
            # Propager avant et après l'événement
            for offset in range(-radius, radius + 1):
                if offset == 0:
                    continue  # Jour même déjà traité
                
                target_idx = event_idx + offset
                if 0 <= target_idx < len(calendar_df):
                    # Impact décroissant avec la distance
                    decay_factor = 1.0 / (abs(offset) + 1)
                    propagated_impact[target_idx] = max(
                        propagated_impact[target_idx],
                        impact * decay_factor
                    )
        
        calendar_df['propagated_impact'] = propagated_impact
        
        # Impact total (original + propagé)
        calendar_df['combined_impact'] = (
            calendar_df['max_impact'] * 0.7 + calendar_df['propagated_impact'] * 0.3
        )
        
        return calendar_df
    
    def _filter_by_date_range(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filtre sur la période configurée."""
        if 'date' not in data.columns:
            return data
        
        start_date = pd.to_datetime(processing_config.start_date)
        end_date = pd.to_datetime(processing_config.end_date)
        
        before = len(data)
        data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        after = len(data)
        
        if before != after:
            self.logger.info(f"   - Filtrage période: {before - after} événements enlevés")
        
        return data
    
    def _log_cleaning_stats(self) -> None:
        """Affiche les statistiques de nettoyage."""
        self.logger.info("📊 Statistiques de nettoyage événements:")
        for key, value in self.stats.items():
            if isinstance(value, (int, float)) and value > 0:
                self.logger.info(f"   - {key}: {value}")