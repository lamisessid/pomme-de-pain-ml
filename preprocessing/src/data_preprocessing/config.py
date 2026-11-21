"""
Configuration centrale du preprocessing.
Tous les chemins, paramètres et constantes sont définis ici.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Chemins de base
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class DataConfig:
    """Configuration des données."""
    
    # Chemins des données brutes
    raw_dir: Path = DATA_DIR / "raw"
    interim_dir: Path = DATA_DIR / "interim"
    processed_dir: Path = DATA_DIR / "processed"
    
    # Fichiers spécifiques (à adapter selon vos noms exacts)
    sales_raw: Path = field(default_factory=lambda: DataConfig.raw_dir / "ventes_journalieres.csv")
    sales_parquet: Path = field(default_factory=lambda: DataConfig.raw_dir / "ventes_journalieres.parquet")
    weather_raw: Path = field(default_factory=lambda: DataConfig.raw_dir / "meteo_sqy.csv")
    events_raw: Path = field(default_factory=lambda: DataConfig.raw_dir / "evenements_sqy.csv")
    events_json: Path = field(default_factory=lambda: DataConfig.raw_dir / "evenements_sqy.json")
    products_raw: Path = field(default_factory=lambda: DataConfig.raw_dir / "produits.csv")
    products_json: Path = field(default_factory=lambda: DataConfig.raw_dir / "produits.json")
    
    # Fichiers intermédiaires
    sales_cleaned: Path = field(default_factory=lambda: DataConfig.interim_dir / "sales_cleaned.parquet")
    weather_cleaned: Path = field(default_factory=lambda: DataConfig.interim_dir / "weather_cleaned.parquet")
    events_cleaned: Path = field(default_factory=lambda: DataConfig.interim_dir / "events_cleaned.parquet")
    products_cleaned: Path = field(default_factory=lambda: DataConfig.interim_dir / "products_cleaned.parquet")
    
    # Fichiers finaux
    final_dataset: Path = field(default_factory=lambda: DataConfig.processed_dir / "final_dataset.parquet")
    train_dataset: Path = field(default_factory=lambda: DataConfig.processed_dir / "train.parquet")
    test_dataset: Path = field(default_factory=lambda: DataConfig.processed_dir / "test.parquet")
    validation_dataset: Path = field(default_factory=lambda: DataConfig.processed_dir / "validation.parquet")
    
    # Colonnes attendues (à adapter après exploration)
    expected_sales_columns: List[str] = field(default_factory=lambda: [
        "date",
        "product_id",
        "quantity",
        "revenue",
        "losses",
        "stock_end"
    ])
    
    expected_weather_columns: List[str] = field(default_factory=lambda: [
        "date",
        "temperature",
        "precipitation",
        "sunshine",
        "humidity"
    ])
    
    expected_events_columns: List[str] = field(default_factory=lambda: [
        "date",
        "event_type",
        "event_name",
        "location",
        "estimated_attendance",
        "impact_level"
    ])
    
    expected_products_columns: List[str] = field(default_factory=lambda: [
        "product_id",
        "name",
        "category",
        "price",
        "cost",
        "shelf_life",
        "popularity",
        "unit"
    ])
    
    def __post_init__(self):
        """Initialise les chemins après la création de l'instance."""
        # Les chemins sont déjà définis avec default_factory, cette méthode
        # est conservée pour d'éventuelles initialisations supplémentaires
        pass


@dataclass
class ProcessingConfig:
    """Configuration du traitement."""
    
    # Dates
    start_date: str = "2022-01-01"
    end_date: str = "2023-12-31"
    
    # Split temporel
    train_end: str = "2023-06-30"  # 18 mois pour train
    validation_end: str = "2023-09-30"  # 3 mois pour validation
    # Le reste (3 mois) pour test
    
    # Nettoyage
    outlier_threshold: float = 3.0  # Écarts standards pour outliers
    min_sales_quantity: int = 0
    max_sales_quantity: int = 1000  # À ajuster
    
    # Weather cleaning
    temp_min: float = -10.0  # °C minimum réaliste pour SQY
    temp_max: float = 40.0   # °C maximum réaliste
    precip_max: float = 100.0  # mm/jour maximum
    
    # Encodage
    categorical_encoding: str = "onehot"  # "onehot" ou "ordinal"
    
    # Features
    lag_features: List[int] = field(default_factory=lambda: [1, 2, 3, 7, 14, 30])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14, 30])


@dataclass
class LoggingConfig:
    """Configuration du logging."""
    
    log_level: str = "INFO"
    log_file: Path = field(default_factory=lambda: PROJECT_ROOT / "logs" / "preprocessing.log")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


# Instances de configuration
data_config = DataConfig()
processing_config = ProcessingConfig()
logging_config = LoggingConfig()


def create_directories():
    """Crée tous les répertoires nécessaires."""
    directories = [
        data_config.raw_dir,
        data_config.interim_dir,
        data_config.processed_dir,
        logging_config.log_file.parent,
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "notebooks",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "scripts",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"✅ Répertoires créés dans {DATA_DIR}")