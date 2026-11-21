"""
Loader pour les données météorologiques.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import logging

from .base_loader import BaseLoader
from ..config import data_config
from ..utils import load_file


class WeatherLoader(BaseLoader):
    """Charge et valide les données météorologiques."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.expected_columns = data_config.expected_weather_columns
        self.primary_key = ['date']
    
    def load(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Charge les données météorologiques.
        """
        self.logger.info("Chargement des données météorologiques...")
        
        # Déterminer le fichier source
        if file_path is None:
            file_path = data_config.weather_raw
        
        if not file_path.exists():
            self.logger.error(f"Fichier météo introuvable: {file_path}")
            raise FileNotFoundError(f"Fichier météo introuvable: {file_path}")
        
        # Charger les données
        self.data = load_file(file_path, self.logger)
        
        # Nettoyer les noms de colonnes
        self._clean_column_names()
        
        # Détecter et convertir les dates
        date_columns = self._detect_date_columns()
        if date_columns:
            self._convert_dates(date_columns)
        else:
            # Essayer de détecter manuellement les dates
            self._manual_date_detection()
        
        # Valider
        if self.validate():
            self.logger.info("✅ Données météo chargées avec succès")
            self._update_metadata()
        else:
            self.logger.warning("⚠️  Données météo chargées avec des problèmes")
        
        return self.data
    
    def _clean_column_names(self) -> None:
        """Nettoie les noms de colonnes météo."""
        if self.data is None:
            return
        
        # Mapping des noms de colonnes courants
        column_mapping = {
    'date': 'date',
    'temperature': 'temperature',
    'pluie_mm': 'precipitation',
    'ensoleillement': 'sunshine',
    'mois': 'month',
    'saison': 'season',
}
        
        # Appliquer le mapping
        new_columns = []
        for col in self.data.columns:
            col_lower = col.strip()
            # Chercher une correspondance exacte ou partielle
            matched = False
            for key, value in column_mapping.items():
                if key.lower() in col_lower.lower():
                    new_columns.append(value)
                    matched = True
                    break
            if not matched:
                new_columns.append(col_lower.replace(' ', '_').lower())
        
        self.data.columns = new_columns
        self.logger.info(f"Colonnes météo après nettoyage: {list(self.data.columns)}")
    
    def _manual_date_detection(self) -> None:
        """Tente de détecter manuellement les colonnes de date."""
        if self.data is None:
            return
        
        # Chercher des colonnes qui pourraient être des dates
        date_candidates = []
        for col in self.data.columns:
            # Noms communs pour les dates
            if any(keyword in col.lower() for keyword in ['date', 'jour', 'time', 'timestamp']):
                date_candidates.append(col)
            
            # Vérifier le type de données
            if pd.api.types.is_datetime64_any_dtype(self.data[col]):
                date_candidates.append(col)
        
        # Essayer de convertir les candidats
        for col in date_candidates:
            try:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce', dayfirst=True)
                if self.data[col].notna().any():
                    self.data = self.data.rename(columns={col: 'date'})
                    self.logger.info(f"✅ Colonne '{col}' convertie en 'date'")
                    break
            except:
                continue
        
        # Si aucune date trouvée, créer une date factice (à adapter selon vos données)
        if 'date' not in self.data.columns:
            self.logger.warning("⚠️  Aucune colonne date trouvée, création d'une date factice")
            # À adapter selon votre période de données
            start_date = pd.Timestamp(data_config.processing_config.start_date)
            self.data['date'] = pd.date_range(
                start=start_date, 
                periods=len(self.data), 
                freq='D'
            )
    
    def validate(self) -> bool:
        """Valide la structure des données météorologiques."""
        if self.data is None:
            self.logger.error("Aucune donnée à valider")
            return False
        
        validation_passed = True
        
        # 1. Vérifier la colonne date
        if 'date' not in self.data.columns:
            self.logger.error("❌ Colonne 'date' manquante")
            validation_passed = False
        else:
            # Vérifier les doublons de dates
            duplicates = self.data['date'].duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"⚠️  {duplicates} dates en doublon dans les données météo")
            
            # Vérifier les dates manquantes
            null_dates = self.data['date'].isnull().sum()
            if null_dates > 0:
                self.logger.warning(f"⚠️  {null_dates} dates manquantes")
            
            # Vérifier la continuité des dates
            if self.data['date'].notna().any():
                self.data = self.data.sort_values('date')
                date_range = pd.date_range(
                    start=self.data['date'].min(),
                    end=self.data['date'].max(),
                    freq='D'
                )
                missing_dates = date_range.difference(self.data['date'])
                if len(missing_dates) > 0:
                    self.logger.warning(f"⚠️  {len(missing_dates)} dates manquantes dans la série")
                    # Option: interpoler les dates manquantes
                    self._fill_missing_dates(missing_dates)
        
        # 2. Vérifier les valeurs aberrantes pour les variables numériques
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col == 'date':
                continue
            
            # Vérifier les valeurs manquantes
            missing = self.data[col].isnull().sum()
            if missing > 0:
                self.logger.info(f"   - {col}: {missing} valeurs manquantes")
            
            # Statistiques de base
            if self.data[col].notna().any():
                stats = self.data[col].describe()
                self.logger.info(f"   - {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
        
        # 3. Vérifier la plage de températures réalistes
        if 'temperature' in self.data.columns:
            unrealistic_temp = self.data[
                (self.data['temperature'] < -20) | (self.data['temperature'] > 45)
            ].shape[0]
            if unrealistic_temp > 0:
                self.logger.warning(f"⚠️  {unrealistic_temp} températures potentiellement irréalistes")
        
        if validation_passed:
            self.logger.info(f"📊 Statistiques météo:")
            self.logger.info(f"   - Nombre de jours: {len(self.data)}")
            self.logger.info(f"   - Période: {self.data['date'].min()} to {self.data['date'].max()}")
            self.logger.info(f"   - Variables disponibles: {list(numeric_cols)}")
        
        return validation_passed
    
    def _fill_missing_dates(self, missing_dates: pd.DatetimeIndex) -> None:
        """Remplit les dates manquantes par interpolation."""
        if self.data is None or len(missing_dates) == 0:
            return
        
        # Créer un DataFrame pour les dates manquantes
        missing_df = pd.DataFrame({'date': missing_dates})
        
        # Fusionner avec les données existantes
        self.data = pd.concat([self.data, missing_df], ignore_index=True)
        self.data = self.data.sort_values('date').reset_index(drop=True)
        
        # Interpoler les valeurs numériques
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col != 'date':
                self.data[col] = self.data[col].interpolate(method='linear')
        
        self.logger.info(f"✅ {len(missing_dates)} dates manquantes interpolées")
    
    def _update_metadata(self) -> None:
        """Met à jour les métadonnées."""
        if self.data is None:
            return
        
        self.metadata.update({
            'rows': len(self.data),
            'date_min': str(self.data['date'].min()) if 'date' in self.data.columns else None,
            'date_max': str(self.data['date'].max()) if 'date' in self.data.columns else None,
            'variables': list(self.data.select_dtypes(include=['number']).columns),
            'missing_values': int(self.data.isnull().sum().sum())
        })
    
    def get_daily_weather(self) -> pd.DataFrame:
        """Retourne les données météo quotidiennes."""
        if self.data is None or 'date' not in self.data.columns:
            return pd.DataFrame()
        
        # S'assurer qu'il n'y a qu'une entrée par jour
        daily_weather = self.data.drop_duplicates(subset=['date'])
        return daily_weather