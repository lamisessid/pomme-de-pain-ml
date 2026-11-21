"""
Règles de commande : commande "siège" (règle historique) et commande recommandée par l'IA.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd

from app.config import settings
from app.data import DataStore


def seat_order(store: DataStore, product_id: int, day: date) -> int:
    """
    Règle du siège : moyenne des ventes des 4 derniers mêmes jours de semaine connus
    (antérieurs à `day` et à la date de référence), majorée d'une marge de sécurité.
    """
    known_until = min(day - timedelta(days=1), store.reference_date)
    series = store.product_series(product_id, end=known_until)["sales"]
    same_weekday = series[series.index.dayofweek == pd.Timestamp(day).dayofweek].tail(4)
    if same_weekday.empty:
        return 0
    return math.ceil(same_weekday.mean() * (1 + settings.SEAT_SAFETY_MARGIN))


def ai_order(predicted_sales: float) -> int:
    return math.ceil(predicted_sales * (1 + settings.AI_SAFETY_MARGIN))
