"""
Catalogue produits et statistiques sur les 30 derniers jours.
"""
from datetime import timedelta
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.data import get_store
from app.services.forecasting import upcoming

router = APIRouter()

PERIOD_DAYS = 30


def _period_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("product_id").agg(
        sales=("sales", "sum"), waste=("waste", "sum"), revenue=("revenue", "sum"), waste_cost=("waste_cost", "sum"))
    g["waste_rate"] = (g["waste"] / g["sales"].where(g["sales"] > 0) * 100).fillna(0)
    return g


@router.get("")
def list_products(category: Optional[str] = None, q: Optional[str] = None):
    store = get_store()
    end = store.reference_date
    cur = _period_stats(store.sales_between(end - timedelta(days=PERIOD_DAYS - 1), end))
    prev = _period_stats(store.sales_between(end - timedelta(days=2 * PERIOD_DAYS - 1), end - timedelta(days=PERIOD_DAYS)))

    products = store.products
    if category:
        products = products[products["category"] == category]
    if q:
        products = products[products["name"].str.contains(q, case=False, regex=False)]

    out = []
    for p in products.itertuples():
        c = cur.loc[p.id]
        prev_sales = prev.loc[p.id, "sales"] if p.id in prev.index else 0
        out.append({
            "id": int(p.id), "name": p.name, "category": p.category, "subcategory": p.subcategory,
            "price": float(p.price), "cost": float(p.cost), "margin_pct": round(float(p.marge_pourcentage), 1),
            "shelf_life_days": int(p.shelf_life_days),
            "sales_30d": int(c["sales"]), "revenue_30d": round(float(c["revenue"]), 2),
            "waste_30d": int(c["waste"]), "waste_rate_30d": round(float(c["waste_rate"]), 2),
            "trend_pct": round(float((c["sales"] - prev_sales) / prev_sales * 100), 1) if prev_sales else None,
        })
    return {"period_days": PERIOD_DAYS, "end": end.isoformat(), "categories": store.categories, "products": out}


@router.get("/{product_id}")
def product_detail(product_id: int):
    store = get_store()
    try:
        product = store.product(product_id)
    except KeyError:
        raise HTTPException(404, "Produit inconnu")

    hist = store.product_series(product_id, end=store.reference_date).tail(60)
    fc = upcoming(7)
    fc = fc[fc["product_id"] == product_id]
    return {
        "product": {
            "id": int(product["id"]), "name": product["name"], "category": product["category"],
            "subcategory": product["subcategory"], "price": float(product["price"]), "cost": float(product["cost"]),
            "shelf_life_days": int(product["shelf_life_days"]),
        },
        "history": [{"date": d.date().isoformat(), "sales": int(r["sales"]), "waste": int(r["waste"])}
                    for d, r in hist.iterrows()],
        "forecast": [
            {"date": r.date.isoformat(), "predicted_sales": round(float(r.sales), 1),
             "lower": round(float(r.lower), 1), "upper": round(float(r.upper), 1),
             "predicted_waste": round(float(r.waste), 1), "risk": r.risk,
             "ai_order": int(r.ai_order), "seat_order": int(r.seat_order)}
            for r in fc.itertuples()
        ],
    }
