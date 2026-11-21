# Preprocessing

Pipeline de préparation du jeu de données simulé Pomme de Pain SQY (2022–2023, 50 produits).

```bash
# depuis la racine du dépôt
make preprocess
```

Étapes (`src/data_preprocessing/main.py`) :

1. **Chargement** (`loaders/`) : ventes journalières par produit, météo, événements SQY, catalogue produits.
2. **Validation** (`validator.py`) : structure et types des colonnes attendues.
3. **Nettoyage** (`cleaners/`) : valeurs manquantes, quantités négatives, outliers, bornes météo réalistes,
   filtrage sur la période d'étude.
4. **Transformations** (`transformers/`) :
   - variables cycliques et saisonnalité ;
   - jours fériés, vacances scolaires et dates spéciales ;
   - lags et moyennes glissantes ;
   - intensité et propagation de l'impact des événements.
5. **Fusion et split temporel** (`merger.py`, `splitter.py`) : train ≤ 2023-06-30, validation T3, test T4.

| Dossier | Contenu | Versionné |
|---|---|---|
| `data/raw/` | Données sources simulées, lues aussi par le backend | oui |
| `data/interim/` | Données nettoyées (parquet) | non, régénéré |
| `data/processed/` | Dataset final et splits (parquet) | non, régénéré |
| `reports/` | Statistiques descriptives et figures d'exploration | oui |

Le modèle du backend part des données brutes (`data/raw`), qui sont complètes (aucun jour manquant), et
construit ses propres features séquentielles (`backend/app/ml/features.py`). La pipeline sert à l'exploration
et au contrôle qualité. Les figures se régénèrent avec `scripts/generate_report_plots.py`.
