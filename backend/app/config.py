"""
Configuration de l'application, lue depuis les variables d'environnement (et un fichier .env optionnel).
"""
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


def _path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


class Settings:
    PROJECT_NAME: str = "Pomme de Pain ML"
    VERSION: str = "2.0.0"

    # Données brutes produites par le module preprocessing
    DATA_DIR: Path = _path("DATA_DIR", BACKEND_DIR.parent / "preprocessing" / "data" / "raw")
    # Modèle entraîné + métriques (générés par scripts/train.py)
    ARTIFACTS_DIR: Path = _path("ARTIFACTS_DIR", BACKEND_DIR / "artifacts")
    # Base SQLite locale (état des alertes)
    DB_PATH: Path = _path("DB_PATH", BACKEND_DIR / "var" / "app.db")

    # Date "aujourd'hui" de la démo. Par défaut : 14 jours avant la fin des données,
    # ce qui permet de comparer les prévisions aux ventes réelles.
    REFERENCE_DATE: Optional[str] = os.getenv("REFERENCE_DATE") or None

    CORS_ORIGINS: List[str] = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:5190,http://127.0.0.1:5190").split(",")
        if o.strip()
    ]

    # Règle de commande du siège : moyenne des 4 mêmes jours de semaine précédents x (1 + marge)
    SEAT_SAFETY_MARGIN: float = float(os.getenv("SEAT_SAFETY_MARGIN", "0.15"))
    # Commande IA : prévision x (1 + marge)
    AI_SAFETY_MARGIN: float = float(os.getenv("AI_SAFETY_MARGIN", "0.05"))

    @property
    def model_path(self) -> Path:
        return self.ARTIFACTS_DIR / "model.pt"

    @property
    def metrics_path(self) -> Path:
        return self.ARTIFACTS_DIR / "metrics.json"


settings = Settings()
