"""
Fonctions utilitaires pour le preprocessing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from tqdm import tqdm
import os


def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Configure un logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler si spécifié
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger


def load_file(file_path: Path, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Charge un fichier selon son extension.
    
    Supporte : .csv, .parquet, .json, .xlsx
    """
    if not file_path.exists():
        error_msg = f"Fichier introuvable : {file_path}"
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, low_memory=False)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            error_msg = f"Format non supporté : {file_path.suffix}"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)
        
        if logger:
            logger.info(f"✅ Fichier chargé : {file_path} - Shape: {df.shape}")
        
        return df
    
    except Exception as e:
        error_msg = f"Erreur lors du chargement de {file_path}: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise


def _clean_dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame pour la sauvegarde en Parquet.
    
    Résout les problèmes de :
    1. Colonnes catégorielles avec des valeurs 'unknown' alors qu'elles devraient être numériques
    2. Types de données incompatibles avec Parquet
    """
    df_clean = df.copy()
    
    # Identifier les colonnes qui pourraient être problématiques
    problematic_cols = []
    
    for col in df_clean.columns:
        # Si c'est une colonne catégorielle
        if pd.api.types.is_categorical_dtype(df_clean[col]):
            # Vérifier si elle contient des valeurs 'unknown' mais devrait être numérique
            categories = list(df_clean[col].cat.categories)
            
            # Vérifier si la colonne devrait être numérique
            # (noms de colonnes qui suggèrent des valeurs numériques)
            numeric_keywords = ['streak', 'count', 'total', 'sum', 'avg', 'mean', 
                               'median', 'std', 'min', 'max', 'lag', 'ma', 'trend',
                               'quantity', 'price', 'cost', 'revenue', 'temperature',
                               'precipitation', 'impact', 'score', 'zscore', 'anomaly']
            
            should_be_numeric = any(keyword in col.lower() for keyword in numeric_keywords)
            
            if should_be_numeric and 'unknown' in categories:
                # Remplacer 'unknown' par une valeur numérique appropriée
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = pd.to_numeric(df_clean[col].replace('unknown', '0'), errors='coerce')
                
                # Si c'est une colonne de comptage, remplir les NaN par 0
                if 'count' in col.lower() or 'streak' in col.lower():
                    df_clean[col] = df_clean[col].fillna(0).astype(int)
                else:
                    df_clean[col] = df_clean[col].fillna(0).astype(float)
                
                problematic_cols.append(col)
        
        # Pour les colonnes d'objet qui devraient être numériques
        elif df_clean[col].dtype == 'object':
            # Essayer de convertir en numérique
            try:
                numeric_version = pd.to_numeric(df_clean[col], errors='coerce')
                if numeric_version.notna().mean() > 0.8:  # Si >80% peuvent être convertis
                    df_clean[col] = numeric_version
                    problematic_cols.append(col)
            except:
                pass
    
    if problematic_cols:
        print(f"⚠️  Colonnes nettoyées pour Parquet: {problematic_cols}")
    
    return df_clean


def save_file(df: pd.DataFrame, file_path: Path, 
              logger: Optional[logging.Logger] = None) -> None:
    """
    Sauvegarde un DataFrame selon l'extension.
    """
    try:
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if file_path.suffix == '.csv':
            df.to_csv(file_path, index=False)
        elif file_path.suffix == '.parquet':
            # Nettoyer le DataFrame avant de sauvegarder en Parquet
            df_clean = _clean_dataframe_for_parquet(df)
            df_clean.to_parquet(file_path, index=False)
        elif file_path.suffix == '.json':
            df.to_json(file_path, orient='records', indent=2)
        else:
            error_msg = f"Format non supporté pour sauvegarde: {file_path.suffix}"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)
        
        if logger:
            logger.info(f"✅ Fichier sauvegardé : {file_path}")
    
    except Exception as e:
        error_msg = f"Erreur lors de la sauvegarde de {file_path}: {str(e)}"
        if logger:
            logger.error(error_msg)
        raise


def detect_date_columns(df: pd.DataFrame, sample_size: int = 1000) -> List[str]:
    """
    Détecte automatiquement les colonnes de dates.
    """
    date_columns = []
    
    for col in df.columns:
        # Essayer de convertir un échantillon
        sample = df[col].dropna().head(sample_size)
        
        if sample.empty:
            continue
        
        # Vérifier si c'est déjà datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_columns.append(col)
            continue
        
        # Essayer de parser comme date
        try:
            # Tester plusieurs formats de date
            test_series = pd.to_datetime(
                sample.astype(str), 
                errors='coerce',
                dayfirst=True,  # Important pour format français
                format='mixed'
            )
            
            # Si plus de 80% des valeurs peuvent être parsées
            if test_series.notna().mean() > 0.8:
                date_columns.append(col)
        
        except:
            continue
    
    return date_columns


def create_calendar_features(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Crée des features temporelles à partir d'une colonne date.
    """
    df = df.copy()
    
    if date_column not in df.columns:
        raise ValueError(f"Colonne {date_column} non trouvée")
    
    # S'assurer que c'est datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Features basiques
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.dayofweek  # 0=lundi, 6=dimanche
    df['day_of_year'] = df[date_column].dt.dayofyear
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    df['quarter'] = df[date_column].dt.quarter
    
    # Features cycliques (pour les modèles ML)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # Features booléennes
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_column].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df[date_column].dt.is_year_start.astype(int)
    df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
    
    return df


def calculate_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les statistiques des valeurs manquantes.
    """
    missing_stats = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum().values,
        'missing_percentage': (df.isnull().sum() / len(df) * 100).values,
        'dtype': df.dtypes.values
    })
    
    missing_stats = missing_stats.sort_values('missing_percentage', ascending=False)
    return missing_stats


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Détecte les outliers avec la méthode IQR.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return {
        'column': column,
        'q1': Q1,
        'q3': Q3,
        'iqr': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outlier_count': len(outliers),
        'outlier_percentage': len(outliers) / len(df) * 100,
        'outliers': outliers[column].tolist()
    }


def print_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Affiche un résumé d'un DataFrame.
    """
    print(f"\n{'='*60}")
    print(f"📊 RÉSUMÉ : {name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Colonnes: {len(df.columns)}")
    print(f"Lignes: {len(df)}")
    
    if 'date' in df.columns:
        print(f"Période: {df['date'].min()} to {df['date'].max()}")
    else:
        print(f"Période: Pas de colonne date")
    
    print(f"\nTypes de données:")
    print(df.dtypes.value_counts())
    
    missing_stats = calculate_missing_stats(df)
    if missing_stats['missing_count'].sum() > 0:
        print(f"\n⚠️  Valeurs manquantes:")
        print(missing_stats[missing_stats['missing_count'] > 0].to_string())
    
    print(f"\nPremières lignes:")
    print(df.head())
    print(f"\nDernières lignes:")
    print(df.tail())


def validate_dataframe(df: pd.DataFrame, required_cols: List[str] = None) -> Dict[str, Any]:
    """
    Valide un DataFrame et retourne des statistiques.
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'stats': {}
    }
    
    # Vérifier la taille
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append("DataFrame vide")
    
    # Vérifier les colonnes requises
    if required_cols:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Colonnes manquantes: {missing_cols}")
    
    # Statistiques
    validation_results['stats']['shape'] = df.shape
    validation_results['stats']['missing_percentage'] = df.isnull().sum().sum() / df.size * 100
    validation_results['stats']['dtypes'] = df.dtypes.value_counts().to_dict()
    
    # Vérifier les valeurs aberrantes pour les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:  # Limiter aux 5 premières pour la performance
        try:
            outliers = detect_outliers_iqr(df, col)
            if outliers['outlier_count'] > 0:
                validation_results['issues'].append(
                    f"{col}: {outliers['outlier_count']} outliers ({outliers['outlier_percentage']:.1f}%)"
                )
        except:
            pass
    
    return validation_results


def get_french_holidays(years: List[int]) -> pd.DataFrame:
    """
    Retourne un DataFrame avec les jours fériés français.
    """
    holidays = []
    
    for year in years:
        # Jours fériés fixes
        fixed_holidays = [
            (f"{year}-01-01", "Jour de l'An"),
            (f"{year}-05-01", "Fête du Travail"),
            (f"{year}-05-08", "Victoire 1945"),
            (f"{year}-07-14", "Fête Nationale"),
            (f"{year}-08-15", "Assomption"),
            (f"{year}-11-01", "Toussaint"),
            (f"{year}-11-11", "Armistice 1918"),
            (f"{year}-12-25", "Noël"),
        ]
        
        # Pâques (dimanche) - calcul simplifié
        # Algorithme de Butcher pour la date de Pâques
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
        
        easter_date = datetime(year, month, day)
        
        # Lundi de Pâques
        easter_monday = easter_date + timedelta(days=1)
        
        # Ascension (39 jours après Pâques)
        ascension = easter_date + timedelta(days=39)
        
        # Lundi de Pentecôte (50 jours après Pâques)
        pentecost_monday = easter_date + timedelta(days=50)
        
        variable_holidays = [
            (easter_monday.strftime("%Y-%m-%d"), "Lundi de Pâques"),
            (ascension.strftime("%Y-%m-%d"), "Ascension"),
            (pentecost_monday.strftime("%Y-%m-%d"), "Lundi de Pentecôte"),
        ]
        
        holidays.extend(fixed_holidays + variable_holidays)
    
    holidays_df = pd.DataFrame(holidays, columns=['date', 'holiday_name'])
    holidays_df['date'] = pd.to_datetime(holidays_df['date'])
    
    return holidays_df


def get_school_holidays(years: List[int], zone: str = "C") -> pd.DataFrame:
    """
    Retourne un DataFrame avec les vacances scolaires pour la zone spécifiée.
    Dates approximatives pour la France.
    """
    all_holidays = []
    
    for year in years:
        # Vacances d'hiver (février)
        winter_start = f"{year}-02-10"
        winter_end = f"{year}-02-25"
        
        # Vacances de printemps (avril)
        spring_start = f"{year}-04-10"
        spring_end = f"{year}-04-25"
        
        # Vacances d'été (juillet-août)
        summer_start = f"{year}-07-05"
        summer_end = f"{year}-08-31"
        
        # Vacances de la Toussaint (octobre-novembre)
        autumn_start = f"{year}-10-20"
        autumn_end = f"{year}-11-05"
        
        # Vacances de Noël (décembre-janvier)
        christmas_start = f"{year}-12-20"
        christmas_end = f"{year+1}-01-05"
        
        holiday_periods = [
            (winter_start, winter_end, "Vacances d'hiver"),
            (spring_start, spring_end, "Vacances de printemps"),
            (summer_start, summer_end, "Vacances d'été"),
            (autumn_start, autumn_end, "Vacances de la Toussaint"),
            (christmas_start, christmas_end, "Vacances de Noël"),
        ]
        
        for start, end, name in holiday_periods:
            date_range = pd.date_range(start=start, end=end, freq='D')
            for date in date_range:
                all_holidays.append((date, name, zone))
    
    holidays_df = pd.DataFrame(all_holidays, columns=['date', 'holiday_type', 'zone'])
    
    return holidays_df