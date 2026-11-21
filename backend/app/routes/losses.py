"""
Analyse des pertes (invendus) sur une période se terminant à la date de référence.
"""
from datetime import timedelta

import numpy as np
from fastapi import APIRouter, Query

from app.data import daily_totals, get_store

router = APIRouter()

WEEKDAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]


def _group(df, by):
    g = df.groupby(by).agg(sales=("sales", "sum"), waste=("waste", "sum"), waste_cost=("waste_cost", "sum"))
    g["waste_rate"] = np.where(g["sales"] > 0, g["waste"] / g["sales"] * 100, 0.0)
    return g.reset_index().round({"waste_cost": 2, "waste_rate": 2})


@router.get("")
def losses(days: int = Query(30, ge=7, le=365)):
    store = get_store()
    end = store.reference_date
    start = end - timedelta(days=days - 1)
    df = store.sales_between(start, end)
    prev = store.sales_between(start - timedelta(days=days), start - timedelta(days=1))

    sales, waste, cost = int(df["sales"].sum()), int(df["waste"].sum()), float(df["waste_cost"].sum())
    prev_cost = float(prev["waste_cost"].sum())

    by_weekday = _group(df.assign(weekday=df["date"].dt.dayofweek), "weekday")
    by_weekday["weekday"] = by_weekday["weekday"].map(lambda i: WEEKDAYS[i])

    df = df.assign(rainy=df["rain"] > 1.0, event=df["has_event"] == 1)
    effects = {}
    for col, labels in (("rainy", ("Sec", "Pluie > 1 mm")), ("event", ("Sans événement", "Avec événement"))):
        g = _group(df, col).set_index(col)
        effects[col] = [
            {"label": labels[int(flag)], "waste_rate": float(g.loc[flag, "waste_rate"]),
             "days": int(df.loc[df[col] == flag, "date"].nunique())}
            for flag in (False, True) if flag in g.index
        ]

    daily = daily_totals(df).reset_index()
    daily["date"] = daily["date"].dt.date.astype(str)

    return {
        "period": {"start": start.isoformat(), "end": end.isoformat(), "days": days},
        "totals": {
            "sales": sales,
            "waste": waste,
            "waste_cost": round(cost, 2),
            "waste_rate": round(waste / sales * 100, 2) if sales else 0.0,
            "avg_daily_cost": round(cost / days, 2),
            "vs_previous_period_pct": round((cost - prev_cost) / prev_cost * 100, 1) if prev_cost else None,
        },
        "by_category": _group(df, "category").sort_values("waste_cost", ascending=False).to_dict("records"),
        "top_products": _group(df, ["product_id", "name", "category"])
            .sort_values("waste_cost", ascending=False).head(10).to_dict("records"),
        "by_weekday": by_weekday.to_dict("records"),
        "effects": effects,
        "daily": daily[["date", "sales", "waste", "waste_cost", "waste_rate"]]
            .round({"waste_cost": 2, "waste_rate": 4}).to_dict("records"),
    }
