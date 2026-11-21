"""
Calendrier des événements SQY et impact mesuré sur les ventes.
"""
import numpy as np
import pandas as pd
from fastapi import APIRouter

from app.data import get_store

router = APIRouter()


@router.get("")
def events():
    store = get_store()
    daily = store.sales.groupby("date")["sales"].sum()
    # Référence : ventes moyennes du même jour de semaine sans événement
    no_event = daily[~daily.index.isin(store.events["date"])]
    baseline = no_event.groupby(no_event.index.dayofweek).mean()

    rows = []
    for e in store.events.sort_values("date").itertuples():
        actual = daily.get(e.date)
        base = baseline.get(e.date.dayofweek)
        rows.append({
            "date": e.date.date().isoformat(), "name": e.name, "type": e.type,
            "expected_impact": float(e.impact), "description": e.description,
            "measured_uplift_pct": round(float((actual - base) / base * 100), 1)
            if actual is not None and base else None,
        })

    by_type = (
        pd.DataFrame(rows).groupby("type")
        .agg(count=("name", "size"), avg_uplift_pct=("measured_uplift_pct", "mean"))
        .reset_index().round(1).replace({np.nan: None}).to_dict("records")
    )
    return {"events": rows, "by_type": by_type}
