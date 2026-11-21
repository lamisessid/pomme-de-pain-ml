"""
Tableau de bord : indicateurs du jour de référence, tendance et prévision du lendemain.
"""
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.data import daily_totals, get_store
from app.services.alerts import generate_alerts
from app.services.forecasting import upcoming

router = APIRouter()


def _pct(current: float, previous: float) -> Optional[float]:
    return round((current - previous) / previous * 100, 1) if previous else None


@router.get("")
def dashboard(day: Optional[date] = None):
    store = get_store()
    day = day or store.reference_date
    if not (store.first_date <= day <= store.last_date):
        raise HTTPException(404, f"Aucune donnée pour le {day}")

    today = store.sales_between(day, day)
    last_week = store.sales_between(day - timedelta(days=7), day - timedelta(days=7))

    def kpis(df):
        sales, waste = float(df["sales"].sum()), float(df["waste"].sum())
        return {
            "sales": int(sales),
            "revenue": round(float(df["revenue"].sum()), 2),
            "waste": int(waste),
            "waste_cost": round(float(df["waste_cost"].sum()), 2),
            "waste_rate": round(waste / sales * 100, 2) if sales else 0.0,
        }

    k, lw = kpis(today), kpis(last_week)
    k["vs_last_week"] = {key: _pct(k[key], lw[key]) for key in ("sales", "revenue", "waste", "waste_cost")}

    top = (
        today.sort_values("revenue", ascending=False).head(5)
        [["product_id", "name", "category", "sales", "revenue"]]
        .rename(columns={"product_id": "id"})
    )

    trend = daily_totals(store.sales_between(day - timedelta(days=29), day)).reset_index()
    trend["date"] = trend["date"].dt.date.astype(str)

    tomorrow = upcoming(1)
    alerts = [a for a in generate_alerts() if not a["resolved"]]

    return {
        "date": day.isoformat(),
        "kpis": k,
        "context": store.context_for(day),
        "top_products": top.round(2).to_dict("records"),
        "trend": trend.round({"revenue": 2, "waste_cost": 2, "waste_rate": 4}).to_dict("records"),
        "tomorrow": {
            "date": (day + timedelta(days=1)).isoformat(),
            "predicted_sales": int(round(tomorrow["sales"].sum())),
            "predicted_revenue": round(float((tomorrow["sales"] * tomorrow["price"]).sum()), 2),
            "predicted_waste": int(round(tomorrow["waste"].sum())),
            "context": store.context_for(day + timedelta(days=1)),
        },
        "alerts": {
            "active": len(alerts),
            "critical": sum(a["severity"] == "critical" for a in alerts),
            "top": alerts[:3],
        },
    }
