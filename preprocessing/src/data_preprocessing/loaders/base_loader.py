"""
Classe de base pour tous les loaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import logging

from ..config import data_config
from ..utils import setup_logger, load_file


class BaseLoader(ABC):
    """Classe abstraite pour le chargement des données."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or setup_logger(self.__class__.__name__)
        self.data = None
        self.metadata = {}
    
    @abstractmethod
    def load(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """Charge les données depuis la source."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Valide la structure des données chargées."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Retourne des informations sur les données chargées."""
        if self.data is None:
            return {"status": "no_data_loaded"}
        
        return {
            "status": "loaded",
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": str(self.data.dtypes.to_dict()),
            "memory_usage": f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            "date_range": {
                "min": str(self.data.get('date', pd.Series()).min()),
                "max": str(self.data.get('date', pd.Series()).max())
            } if 'date' in self.data.columns else None
        }
    
    def save(self, file_path: Path) -> None:
        """Sauvegarde les données chargées."""
        if self.data is None:
            self.logger.warning("Aucune donnée à sauvegarder")
            return
        
        from ..utils import save_file
        save_file(self.data, file_path, self.logger)
    
    def _detect_date_columns(self) -> list:
        """Détecte les colonnes de date."""
        from ..utils import detect_date_columns
        return detect_date_columns(self.data)
    
    def _convert_dates(self, date_columns: list) -> None:
        """Convertit les colonnes de date en datetime."""
        if not date_columns:
            return
        
        for col in date_columns:
            try:
                self.data[col] = pd.to_datetime(
                    self.data[col], 
                    errors='coerce',
                    dayfirst=True  # Format français
                )
                self.logger.info(f"✅ Converti {col} en datetime")
            except Exception as e:
                self.logger.warning(f"Échec conversion de {col}: {e}")