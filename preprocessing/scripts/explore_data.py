# scripts/explore_data.py
#!/usr/bin/env python3
"""
Script pour explorer la structure des données.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

def explore_sales():
    """Explore les données de ventes."""
    print("🔍 EXPLORATION DES VENTES")
    print("="*60)
    
    df = pd.read_parquet("data/raw/ventes_journalieres.parquet")
    
    print(f"Shape: {df.shape}")
    print(f"\nColonnes: {list(df.columns)}")
    print(f"\nTypes:")
    print(df.dtypes)
    
    print(f"\nValeurs manquantes:")
    print(df.isnull().sum())
    
    print(f"\nDates uniques: {df['date'].nunique()}")
    print(f"Produits uniques: {df['produit_id'].nunique()}")
    
    # Vérifier les dates manquantes
    date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    dates_in_data = pd.to_datetime(df['date'].dropna().unique())
    missing_dates = date_range.difference(dates_in_data)
    
    print(f"\n📅 Dates manquantes: {len(missing_dates)}")
    if len(missing_dates) > 0:
        print(f"Exemples: {missing_dates[:10].tolist()}")
    
    # Échantillon
    print(f"\n📋 Échantillon (5 premières lignes):")
    print(df.head())
    
    return df

def explore_weather():
    """Explore les données météo."""
    print("\n🔍 EXPLORATION MÉTÉO")
    print("="*60)
    
    df = pd.read_csv("data/raw/meteo_sqy.csv")
    
    print(f"Shape: {df.shape}")
    print(f"\nColonnes: {list(df.columns)}")
    print(f"\nTypes:")
    print(df.dtypes)
    
    print(f"\nValeurs manquantes:")
    print(df.isnull().sum())
    
    print(f"\n📋 Échantillon:")
    print(df.head())
    
    return df

def explore_products():
    """Explore les données produits."""
    print("\n🔍 EXPLORATION PRODUITS")
    print("="*60)
    
    df = pd.read_csv("data/raw/produits.csv")
    
    print(f"Shape: {df.shape}")
    print(f"\nColonnes: {list(df.columns)}")
    print(f"\nTypes:")
    print(df.dtypes)
    
    print(f"\nValeurs manquantes:")
    print(df.isnull().sum())
    
    print(f"\n📋 Échantillon:")
    print(df.head())
    
    # Statistiques prix
    print(f"\n💰 Statistiques prix:")
    print(f"  Min: {df['prix'].min():.2f}€")
    print(f"  Max: {df['prix'].max():.2f}€")
    print(f"  Moyenne: {df['prix'].mean():.2f}€")
    print(f"  Médiane: {df['prix'].median():.2f}€")

    return df

def main():
    """Fonction principale."""
    print("🚀 Exploration des données Pomme de Pain")
    print("="*60)
    
    # Explorez chaque type de données
    sales_df = explore_sales()
    weather_df = explore_weather()
    products_df = explore_products()
    
    print("\n✅ Exploration terminée")

if __name__ == "__main__":
    main()