"""
Prévisions de demande (LSTM), métriques d'évaluation et backtest.
"""
import json
from datetime import timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.data import get_store
from app.services.forecasting import backtest_window, require_predictor, upcoming

router = APIRouter()


def _num(v):
    return None if pd.isna(v) else round(float(v), 1)


@router.get("")
def forecasts(days: int = Query(7, ge=1, le=14), product_id: Optional[int] = None):
    store = get_store()
    df = upcoming(days)
    if product_id is not None:
        if product_id not in store.products.index:
            raise HTTPException(404, "Produit inconnu")
        df = df[df["product_id"] == product_id]

    daily = df.groupby("date").agg(
        predicted_sales=("sales", "sum"), lower=("lower", "sum"), upper=("upper", "sum"),
        predicted_waste=("waste", "sum"), actual_sales=("actual_sales", lambda s: s.sum(min_count=1)),
        ai_order=("ai_order", "sum"), seat_order=("seat_order", "sum"),
    ).reset_index()

    history = []
    if product_id is not None:
        hist = store.product_series(product_id, end=store.reference_date).tail(28)
        history = [{"date": d.date().isoformat(), "sales": int(r["sales"]), "waste": int(r["waste"])}
                   for d, r in hist.iterrows()]

    by_product = (
        df.groupby(["product_id", "name", "category"]).agg(
            predicted_sales=("sales", "sum"), predicted_waste=("waste", "sum"),
            ai_order=("ai_order", "sum"), seat_order=("seat_order", "sum"),
            high_risk_days=("risk", lambda s: int((s == "high").sum())),
        ).reset_index().sort_values("predicted_sales", ascending=False)
    )

    return {
        "reference_date": store.reference_date.isoformat(),
        "start_date": daily["date"].min().isoformat(),
        "end_date": daily["date"].max().isoformat(),
        "days": days,
        "daily": [
            {
                "date": r.date.isoformat(),
                "predicted_sales": _num(r.predicted_sales),
                "lower": _num(r.lower),
                "upper": _num(r.upper),
                "predicted_waste": _num(r.predicted_waste),
                "actual_sales": _num(r.actual_sales),
                "ai_order": int(r.ai_order),
                "seat_order": int(r.seat_order),
                "context": {k: v for k, v in store.context_for(r.date).items() if k != "source"},
            }
            for r in daily.itertuples()
        ],
        "history": history,
        "by_product": by_product.round(1).to_dict("records"),
        "summary": {
            "predicted_sales": int(round(df["sales"].sum())),
            "predicted_revenue": round(float((df["sales"] * df["price"]).sum()), 2),
            "predicted_waste": int(round(df["waste"].sum())),
            "predicted_waste_cost": round(float((df["waste"] * df["cost"]).sum()), 2),
        },
    }


@router.get("/accuracy")
def accuracy():
    require_predictor()
    return json.loads(settings.metrics_path.read_text())


@router.get("/backtest")
def backtest(days: int = Query(30, ge=7, le=180), product_id: Optional[int] = None):
    """Prévisions J+1 passées comparées aux ventes réelles et à la règle du siège."""
    df = backtest_window(days)
    if product_id is not None:
        df = df[df["product_id"] == product_id]
    daily = df.groupby("date").agg(
        actual_sales=("actual_sales", "sum"), predicted_sales=("pred_sales", "sum"),
        seat_forecast=("seat_mean4", "sum"),
    ).reset_index()
    abs_err = (df["pred_sales"] - df["actual_sales"]).abs().sum()
    seat_err = (df["seat_mean4"] - df["actual_sales"]).abs().sum()
    total = df["actual_sales"].sum()
    return {
        "days": days,
        "daily": [
            {"date": r.date.date().isoformat(), "actual_sales": round(float(r.actual_sales), 1),
             "predicted_sales": round(float(r.predicted_sales), 1), "seat_forecast": round(float(r.seat_forecast), 1)}
            for r in daily.itertuples()
        ],
        "wape": {
            "lstm": round(float(abs_err / total * 100), 2) if total else None,
            "moyenne_4_semaines": round(float(seat_err / total * 100), 2) if total else None,
        },
    }
