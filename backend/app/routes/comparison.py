"""
Commande siège vs commande IA : simulation sur l'historique récent et recommandations du lendemain.
"""
import numpy as np
from fastapi import APIRouter, Query

from app.config import settings
from app.data import get_store
from app.services.forecasting import backtest_window, upcoming

router = APIRouter()


def _simulate(df, order_col):
    over = np.maximum(df[order_col] - df["actual_sales"], 0)
    under = np.maximum(df["actual_sales"] - df[order_col], 0)
    return df.assign(
        unsold=over, missed=under,
        unsold_cost=over * df["cost"], missed_margin=under * (df["price"] - df["cost"]),
    )


def _totals(df):
    return {
        "units_ordered": int(df["order"].sum()),
        "unsold_units": int(df["unsold"].sum()),
        "unsold_cost": round(float(df["unsold_cost"].sum()), 2),
        "missed_units": int(df["missed"].sum()),
        "missed_margin": round(float(df["missed_margin"].sum()), 2),
        "total_cost": round(float(df["unsold_cost"].sum() + df["missed_margin"].sum()), 2),
    }


@router.get("")
def comparison(days: int = Query(30, ge=7, le=180)):
    store = get_store()
    bt = backtest_window(days)
    bt = bt.assign(
        ai=np.ceil(bt["pred_sales"] * (1 + settings.AI_SAFETY_MARGIN)),
        seat=np.ceil(bt["seat_mean4"] * (1 + settings.SEAT_SAFETY_MARGIN)),
    )
    ai = _simulate(bt.assign(order=bt["ai"]), "order")
    seat = _simulate(bt.assign(order=bt["seat"]), "order")
    t_ai, t_seat = _totals(ai), _totals(seat)

    per_product = []
    for (pid, name, cat), g_ai in ai.groupby(["product_id", "name", "category"]):
        g_seat = seat[seat["product_id"] == pid]
        a, s = _totals(g_ai), _totals(g_seat)
        per_product.append({
            "product_id": int(pid), "name": name, "category": cat,
            "actual_sales": int(g_ai["actual_sales"].sum()),
            "ai": a, "seat": s, "savings": round(s["total_cost"] - a["total_cost"], 2),
        })
    per_product.sort(key=lambda p: p["savings"], reverse=True)

    daily = []
    for d, g in ai.groupby("date"):
        gs = seat[seat["date"] == d]
        daily.append({
            "date": d.date().isoformat(),
            "ai_cost": round(float(g["unsold_cost"].sum() + g["missed_margin"].sum()), 2),
            "seat_cost": round(float(gs["unsold_cost"].sum() + gs["missed_margin"].sum()), 2),
        })

    tomorrow = upcoming(1)
    recommendations = [
        {
            "product_id": int(r.product_id), "name": r.name, "category": r.category,
            "predicted_sales": round(float(r.sales), 1),
            "ai_order": int(r.ai_order), "seat_order": int(r.seat_order),
            "difference": int(r.ai_order - r.seat_order),
            "action": "augmenter" if r.ai_order > r.seat_order else "réduire" if r.ai_order < r.seat_order else "maintenir",
        }
        for r in tomorrow.itertuples()
    ]
    recommendations.sort(key=lambda r: abs(r["difference"]), reverse=True)

    return {
        "period": {"days": days, "end": store.reference_date.isoformat()},
        "rules": {
            "ai": f"prévision LSTM J+1 × {1 + settings.AI_SAFETY_MARGIN:.2f}",
            "seat": f"moyenne des 4 mêmes jours de semaine × {1 + settings.SEAT_SAFETY_MARGIN:.2f}",
        },
        "totals": {"ai": t_ai, "seat": t_seat, "savings": round(t_seat["total_cost"] - t_ai["total_cost"], 2)},
        "per_product": per_product,
        "daily": daily,
        "tomorrow": {"date": tomorrow["date"].iloc[0].isoformat(), "recommendations": recommendations},
    }
