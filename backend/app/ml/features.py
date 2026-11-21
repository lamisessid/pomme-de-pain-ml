"""
Construction des features du modèle.

Ce module est utilisé à la fois par l'entraînement (scripts/train.py) et par l'inférence
(app/ml/predictor.py) : c'est la seule définition des features, ce qui garantit que le
modèle reçoit en production exactement ce qu'il a vu à l'entraînement.

Pour un jour cible t, la séquence d'entrée couvre les jours t-LOOKBACK+1 .. t.
La ligne du jour s contient le contexte du jour s (calendrier, météo, événement) et
l'historique connu la veille (ventes/pertes de s-1, s-7...). Aucune information du jour
cible n'est donc utilisée pour prédire ses ventes.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

LOOKBACK = 28
MAX_LAG = 7
# Premier indice de jour pour lequel une séquence complète existe
MIN_TARGET_INDEX = LOOKBACK + MAX_LAG - 1

DYNAMIC_FEATURES = [
    "lag1_sales", "lag1_waste", "lag7_sales", "mean7_sales",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    "is_weekend", "is_holiday", "temperature", "rain",
    "has_event", "event_impact",
]

# Seuils de taux de perte (pertes / ventes) pour la classe de risque
RISK_THRESHOLDS = (0.05, 0.12)
RISK_LABELS = ["low", "medium", "high"]


def feature_names(categories: Sequence[str]) -> List[str]:
    return DYNAMIC_FEATURES + [f"cat_{c}" for c in categories] + ["price", "shelf_life"]


def static_features(product: dict, categories: Sequence[str]) -> np.ndarray:
    onehot = [1.0 if product["category"] == c else 0.0 for c in categories]
    return np.array(
        onehot + [product["price"] / 10.0, min(product["shelf_life_days"], 30) / 7.0],
        dtype=np.float32,
    )


def build_frame(
    sales: np.ndarray,
    waste: np.ndarray,
    dates: pd.DatetimeIndex,
    context: pd.DataFrame,
    scale: float,
    static: np.ndarray,
) -> np.ndarray:
    """
    Matrice (n_jours, n_features) pour une série produit continue.

    sales/waste : historiques bruts (unités), alignés sur `dates`.
    context     : colonnes temperature, rain, is_holiday, has_event, event_impact, alignées sur `dates`.
    scale       : normalisation propre au produit (ventes moyennes à l'entraînement).
    """
    n = len(dates)
    s = sales.astype(np.float32) / scale
    w = waste.astype(np.float32) / scale

    lag1_s = np.concatenate([[s[0]], s[:-1]])
    lag1_w = np.concatenate([[w[0]], w[:-1]])
    lag7_s = np.concatenate([np.full(MAX_LAG, s[0]), s[:-MAX_LAG]])[:n]
    csum = np.concatenate([[0.0], np.cumsum(s)])
    idx = np.arange(n)
    lo = np.maximum(idx - MAX_LAG, 0)
    mean7 = np.where(idx > 0, (csum[idx] - csum[lo]) / np.maximum(idx - lo, 1), s[0])

    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy()
    dyn = np.column_stack([
        lag1_s, lag1_w, lag7_s, mean7,
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        (dow >= 5).astype(np.float32),
        context["is_holiday"].to_numpy(dtype=np.float32),
        context["temperature"].to_numpy(dtype=np.float32) / 30.0,
        np.clip(context["rain"].to_numpy(dtype=np.float32) / 10.0, 0, 2),
        context["has_event"].to_numpy(dtype=np.float32),
        context["event_impact"].to_numpy(dtype=np.float32) - 1.0,
    ]).astype(np.float32)
    return np.hstack([dyn, np.tile(static, (n, 1))]).astype(np.float32)


def risk_class(sales: np.ndarray, waste: np.ndarray) -> np.ndarray:
    rate = np.where(sales > 0, waste / np.maximum(sales, 1e-6), 1.0)
    return np.digitize(rate, RISK_THRESHOLDS).astype(np.int64)
