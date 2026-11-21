"""
Tables de prévision enrichies (produit, commandes IA / siège, ventes réelles si connues).
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import pandas as pd
from fastapi import HTTPException

from app.data import get_store
from app.ml.predictor import Predictor, get_predictor
from app.services.orders import ai_order, seat_order


def require_predictor() -> Predictor:
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(503, "Modèle non entraîné : lancez `python -m scripts.train` dans backend/.")
    return predictor


@lru_cache(maxsize=16)
def forecast_table(start: date, horizon: int) -> pd.DataFrame:
    store = get_store()
    df = require_predictor().forecast(start, horizon)
    products = store.products[["id", "name", "category", "price", "cost"]]
    df = df.merge(products, left_on="product_id", right_on="id").drop(columns="id")
    df["ai_order"] = df["sales"].map(ai_order)
    df["seat_order"] = [seat_order(store, int(p), d) for p, d in zip(df["product_id"], df["date"])]

    actual = store.sales[["date", "product_id", "sales", "waste"]].rename(
        columns={"sales": "actual_sales", "waste": "actual_waste"})
    actual["date"] = actual["date"].dt.date
    return df.merge(actual, on=["date", "product_id"], how="left")


def upcoming(horizon: int = 7) -> pd.DataFrame:
    """Prévisions à partir du lendemain de la date de référence."""
    return forecast_table(get_store().reference_date + timedelta(days=1), horizon)


def backtest_window(days: int) -> pd.DataFrame:
    """Backtest J+1 sur les `days` jours se terminant à la date de référence."""
    store = get_store()
    bt = require_predictor().backtest
    end = pd.Timestamp(store.reference_date)
    out = bt[(bt["date"] > end - pd.Timedelta(days=days)) & (bt["date"] <= end)]
    return out.merge(store.products[["id", "name", "category", "price", "cost"]],
                     left_on="product_id", right_on="id").drop(columns="id")
