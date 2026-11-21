# Backend — API FastAPI et modèle LSTM

```bash
# depuis la racine du dépôt
make setup
make api            # http://localhost:8642/docs
make test
make train          # régénère artifacts/model.pt et artifacts/metrics.json
```

Sans `make` :

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8642
```

## Organisation

| Fichier | Rôle |
|---|---|
| `app/data.py` | Chargement des CSV de `preprocessing/data/raw`, contexte d'un jour (météo, férié, événements) |
| `app/ml/features.py` | Définition des features, partagée par l'entraînement et l'inférence |
| `app/ml/dataset.py` | Construction des séquences (X, y) et des baselines |
| `app/ml/model.py` | `MultiTaskLSTM` : LSTM + attention + têtes ventes / pertes / risque |
| `app/ml/predictor.py` | Chargement du checkpoint, prévision récursive sur N jours, backtest J+1 |
| `app/services/` | Règles de commande (siège / IA), alertes, tables de prévision mises en cache |
| `app/routes/` | Endpoints REST |
| `app/storage.py` | SQLite (`var/app.db`) : alertes résolues |
| `scripts/train.py` | Entraînement, early stopping, évaluation sur le test, écriture des artefacts |

## Configuration

Toutes les variables sont optionnelles, voir `.env.example`. Les principales sont `REFERENCE_DATE` (date
« aujourd'hui » de la démo) et `CORS_ORIGINS`.

## Découpage temporel

| Jeu | Période |
|---|---|
| Entraînement | 2022-01-01 → 2023-06-30 |
| Validation (early stopping, calibration de l'intervalle) | 2023-07-01 → 2023-09-30 |
| Test (métriques publiées) | 2023-10-01 → 2023-12-31 |

Un test (`tests/test_features.py`) vérifie qu'aucune information du jour cible n'entre dans ses propres features.
