"""
Loader pour les données d'événements.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import json
import logging

from .base_loader import BaseLoader
from ..config import data_config
from ..utils import load_file


class EventsLoader(BaseLoader):
    """Charge et valide les données d'événements."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.expected_columns = data_config.expected_events_columns
        self.primary_key = ['date', 'event_name']
        
        # Catégories d'événements attendues pour SQY
        self.expected_event_types = [
            'match',            # Matchs de sport
            'concert',          # Concerts
            'festival',         # Festivals
            'conference',       # Conférences
            'exhibition',       # Expositions
            'university_event', # Événements universitaires (UVSQ)
            'market',           # Marchés
            'demonstration',    # Manifestations
            'construction',     # Travaux
            'transport_strike', # Grèves transports
            'other'             # Autres
        ]
    
    def load(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Charge les données d'événements.
        
        Stratégie:
        1. Essayer le CSV d'abord
        2. Sinon essayer le JSON
        3. Standardiser le format
        """
        self.logger.info("Chargement des données d'événements...")
        
        # Déterminer le fichier source
        if file_path is None:
            # Essayer CSV d'abord
            if data_config.events_raw.exists():
                file_path = data_config.events_raw
                self.logger.info("Utilisation du fichier CSV")
            elif data_config.events_json.exists():
                file_path = data_config.events_json
                self.logger.info("Utilisation du fichier JSON")
            else:
                self.logger.warning("⚠️  Aucun fichier d'événements trouvé, création d'un DataFrame vide")
                self.data = pd.DataFrame(columns=self.expected_columns)
                return self.data
        
        # Charger les données selon le format
        if file_path.suffix == '.json':
            self.data = self._load_json(file_path)
        else:
            self.data = load_file(file_path, self.logger)
        
        # Nettoyer les noms de colonnes
        self._clean_column_names()
        
        # Détecter et convertir les dates
        date_columns = self._detect_date_columns()
        if date_columns:
            self._convert_dates(date_columns)
        else:
            self._manual_date_detection()
        
        # Valider et standardiser
        if self.validate():
            self.logger.info("✅ Données d'événements chargées avec succès")
            self._standardize_event_types()
            self._update_metadata()
        else:
            self.logger.warning("⚠️  Données d'événements chargées avec des problèmes")
        
        return self.data
    
    def _load_json(self, file_path: Path) -> pd.DataFrame:
        """Charge les données depuis un fichier JSON."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convertir en DataFrame
            # Gérer différents formats JSON possibles
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Si c'est un dict avec une clé contenant la liste
                for key, value in data.items():
                    if isinstance(value, list):
                        df = pd.DataFrame(value)
                        break
                else:
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame()
            
            self.logger.info(f"✅ JSON chargé - {len(df)} événements")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du chargement JSON: {e}")
            return pd.DataFrame()
    
    def _clean_column_names(self) -> None:
        """Nettoie les noms de colonnes des événements."""
        if self.data is None or self.data.empty:
            return
        
        column_mapping = {
            'Date': 'date',
            'DATE': 'date',
            'Jour': 'date',
            'Événement': 'event_name',
            'Event': 'event_name',
            'Nom': 'event_name',
            'Type': 'event_type',
            'Catégorie': 'event_type',
            'Category': 'event_type',
            'Lieu': 'location',
            'Place': 'location',
            'Location': 'location',
            'Affluence': 'estimated_attendance',
            'Attendance': 'estimated_attendance',
            'Participants': 'estimated_attendance',
            'Impact': 'impact_level',
            'Effet': 'impact_level',
            'Influence': 'impact_level',
            'Description': 'description',
            'Details': 'description',
            'Détails': 'description',
        }
        
        # Appliquer le mapping
        new_columns = []
        for col in self.data.columns:
            col_clean = col.strip()
            if col_clean in column_mapping:
                new_columns.append(column_mapping[col_clean])
            else:
                # Standardiser: minuscules, underscores
                new_columns.append(
                    col_clean.lower()
                    .replace(' ', '_')
                    .replace('-', '_')
                    .replace('é', 'e')
                    .replace('è', 'e')
                    .replace('ê', 'e')
                )
        
        self.data.columns = new_columns
        self.logger.info(f"Colonnes événements après nettoyage: {list(self.data.columns)}")
    
    def _manual_date_detection(self) -> None:
        """Tente de détecter manuellement les colonnes de date."""
        if self.data is None or self.data.empty:
            return
        
        # Chercher des colonnes qui pourraient être des dates
        for col in self.data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'jour', 'time', 'when']):
                try:
                    self.data[col] = pd.to_datetime(self.data[col], errors='coerce', dayfirst=True)
                    if self.data[col].notna().any():
                        self.data = self.data.rename(columns={col: 'date'})
                        self.logger.info(f"✅ Colonne '{col}' convertie en 'date'")
                        break
                except:
                    continue
        
        # Si aucune date trouvée, essayer de parser depuis d'autres colonnes
        if 'date' not in self.data.columns:
            for col in self.data.columns:
                if 'description' in col.lower() or 'name' in col.lower():
                    # Essayer d'extraire des dates du texte
                    try:
                        # Pattern simple pour dates dans le texte
                        date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
                        extracted_dates = self.data[col].str.extract(date_pattern)[0]
                        if extracted_dates.notna().any():
                            self.data['date'] = pd.to_datetime(extracted_dates, errors='coerce', dayfirst=True)
                            self.logger.info(f"✅ Dates extraites de la colonne '{col}'")
                            break
                    except:
                        continue
    
    def validate(self) -> bool:
        """Valide la structure des données d'événements."""
        if self.data is None or self.data.empty:
            self.logger.warning("⚠️  Aucune donnée d'événement à valider")
            return True  # Vide mais valide
        
        validation_passed = True
        
        # 1. Vérifier les colonnes critiques
        if 'event_name' not in self.data.columns:
            self.logger.warning("⚠️  Colonne 'event_name' manquante, création depuis index")
            self.data['event_name'] = [f"event_{i}" for i in range(len(self.data))]
        
        # 2. Vérifier la colonne date
        if 'date' not in self.data.columns:
            self.logger.error("❌ Colonne 'date' manquante - nécessaire pour l'analyse")
            validation_passed = False
        else:
            null_dates = self.data['date'].isnull().sum()
            if null_dates > 0:
                self.logger.warning(f"⚠️  {null_dates} dates manquantes dans les événements")
                # Supprimer les événements sans date
                self.data = self.data.dropna(subset=['date'])
        
        # 3. Vérifier/créer la colonne event_type
        if 'event_type' not in self.data.columns:
            self.logger.info("Création automatique des types d'événements...")
            self._infer_event_types()
        
        # 4. Normaliser l'impact si présent
        if 'impact_level' in self.data.columns:
            self._normalize_impact_levels()
        
        # 5. Statistiques
        if validation_passed:
            self.logger.info(f"📊 Statistiques événements:")
            self.logger.info(f"   - Nombre d'événements: {len(self.data)}")
            self.logger.info(f"   - Période: {self.data['date'].min()} to {self.data['date'].max()}")
            
            if 'event_type' in self.data.columns:
                type_counts = self.data['event_type'].value_counts()
                self.logger.info(f"   - Types d'événements: {type_counts.to_dict()}")
        
        return validation_passed
    
    def _infer_event_types(self) -> None:
        """Infère le type d'événement à partir du nom ou de la description."""
        if self.data is None:
            return
        
        # Mots-clés pour chaque type d'événement
        event_keywords = {
            'match': ['match', 'football', 'rugby', 'sport', 'stade', 'championnat'],
            'concert': ['concert', 'musique', 'groupe', 'artist', 'live'],
            'festival': ['festival', 'fête', 'carnaval', 'celebration'],
            'conference': ['conférence', 'colloque', 'séminaire', 'congrès'],
            'exhibition': ['exposition', 'expo', 'salon', 'foire'],
            'university_event': ['université', 'uvsq', 'campus', 'étudiant', 'exam'],
            'market': ['marché', 'brocante', 'vide-grenier'],
            'demonstration': ['manifestation', 'grève', 'protest', 'cortège'],
            'construction': ['travaux', 'chantier', 'fermeture', 'déviation'],
            'transport_strike': ['grève sncf', 'rer c', 'transport', 'mouvement'],
        }
        
        def infer_type(row):
            # Chercher dans toutes les colonnes textuelles
            text_fields = []
            for col in self.data.columns:
                if col != 'date' and pd.api.types.is_string_dtype(self.data[col]):
                    if pd.notna(row[col]):
                        text_fields.append(str(row[col]).lower())
            
            combined_text = ' '.join(text_fields)
            
            # Chercher les mots-clés
            for event_type, keywords in event_keywords.items():
                for keyword in keywords:
                    if keyword in combined_text:
                        return event_type
            
            return 'other'
        
        self.data['event_type'] = self.data.apply(infer_type, axis=1)
        self.logger.info(f"Types d'événements inférés: {self.data['event_type'].value_counts().to_dict()}")
    
    def _normalize_impact_levels(self) -> None:
        """Normalise les niveaux d'impact."""
        if self.data is None or 'impact_level' not in self.data.columns:
            return
        
        # Mapping des valeurs d'impact
        impact_mapping = {
            'faible': 'low',
            'low': 'low',
            'bas': 'low',
            '1': 'low',
            'moyen': 'medium',
            'medium': 'medium',
            'modéré': 'medium',
            '2': 'medium',
            'fort': 'high',
            'high': 'high',
            'élevé': 'high',
            '3': 'high',
            'très fort': 'very_high',
            'very high': 'very_high',
            '4': 'very_high',
        }
        
        # Normaliser
        self.data['impact_level'] = (
            self.data['impact_level']
            .astype(str)
            .str.lower()
            .str.strip()
            .map(impact_mapping)
            .fillna('unknown')
        )
        
        self.logger.info(f"Niveaux d'impact: {self.data['impact_level'].value_counts().to_dict()}")
    
    def _standardize_event_types(self) -> None:
        """Standardise les types d'événements."""
        if self.data is None or 'event_type' not in self.data.columns:
            return
        
        # Mapping des types
        type_mapping = {
            'sport': 'match',
            'football': 'match',
            'rugby': 'match',
            'basket': 'match',
            'musique': 'concert',
            'show': 'concert',
            'spectacle': 'concert',
            'fête': 'festival',
            'carnaval': 'festival',
            'conf': 'conference',
            'séminaire': 'conference',
            'expo': 'exhibition',
            'salon': 'exhibition',
            'universitaire': 'university_event',
            'étudiant': 'university_event',
            'exam': 'university_event',
            'marché': 'market',
            'brocante': 'market',
            'manifestation': 'demonstration',
            'grève': 'transport_strike' if self._is_transport_strike else 'demonstration',
            'travaux': 'construction',
            'chantier': 'construction',
        }
        
        # Standardiser
        self.data['event_type'] = (
            self.data['event_type']
            .astype(str)
            .str.lower()
            .str.strip()
            .map(lambda x: type_mapping.get(x, x))
        )
        
        # S'assurer que tous les types sont dans la liste attendue
        unknown_types = set(self.data['event_type'].unique()) - set(self.expected_event_types)
        if unknown_types:
            self.logger.warning(f"⚠️  Types d'événements non standardisés: {unknown_types}")
            # Convertir les types inconnus en 'other'
            self.data['event_type'] = self.data['event_type'].apply(
                lambda x: x if x in self.expected_event_types else 'other'
            )
    
    def _is_transport_strike(self, event_name: str) -> bool:
        """Détermine si un événement est une grève des transports."""
        if not isinstance(event_name, str):
            return False
        
        transport_keywords = ['rer', 'sncf', 'transport', 'bus', 'métro', 'train']
        strike_keywords = ['grève', 'mouvement', 'perturbation']
        
        name_lower = event_name.lower()
        return (
            any(keyword in name_lower for keyword in transport_keywords) and
            any(keyword in name_lower for keyword in strike_keywords)
        )
    
    def _update_metadata(self) -> None:
        """Met à jour les métadonnées."""
        if self.data is None or self.data.empty:
            self.metadata.update({
                'status': 'empty',
                'event_count': 0
            })
            return
        
        self.metadata.update({
            'event_count': len(self.data),
            'date_min': str(self.data['date'].min()) if 'date' in self.data.columns else None,
            'date_max': str(self.data['date'].max()) if 'date' in self.data.columns else None,
            'event_types': self.data['event_type'].value_counts().to_dict() if 'event_type' in self.data.columns else {},
            'unique_locations': self.data['location'].nunique() if 'location' in self.data.columns else 0
        })
    
    def get_events_by_date(self, target_date: pd.Timestamp) -> pd.DataFrame:
        """Récupère tous les événements pour une date donnée."""
        if self.data is None or 'date' not in self.data.columns:
            return pd.DataFrame()
        
        return self.data[self.data['date'] == target_date].copy()
    
    def get_events_by_type(self, event_type: str) -> pd.DataFrame:
        """Récupère tous les événements d'un type donné."""
        if self.data is None or 'event_type' not in self.data.columns:
            return pd.DataFrame()
        
        return self.data[self.data['event_type'] == event_type].copy()