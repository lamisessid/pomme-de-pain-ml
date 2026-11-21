"""
Génération des alertes à partir des prévisions des prochains jours.

- rupture     : la commande du siège est sous la borne basse de la prévision
- gaspillage  : la commande du siège dépasse la borne haute de la prévision
- evenement   : événement SQY prévu dans l'horizon
"""
from __future__ import annotations

from datetime import timedelta
from typing import Dict, List

from app.data import get_store
from app.services.forecasting import upcoming
from app import storage

HORIZON_DAYS = 3
MIN_IMPACT_EUR = 5.0


def _severity(impact: float) -> str:
    if impact >= 40:
        return "critical"
    if impact >= 20:
        return "high"
    if impact >= 8:
        return "medium"
    return "low"


def generate_alerts() -> List[Dict]:
    store = get_store()
    df = upcoming(HORIZON_DAYS)
    alerts: List[Dict] = []

    for r in df.itertuples():
        day = r.date.isoformat()
        if r.seat_order < r.lower:
            missing = r.sales - r.seat_order
            impact = missing * (r.price - r.cost)
            if impact >= MIN_IMPACT_EUR:
                alerts.append({
                    "id": f"rupture-{r.product_id}-{day}",
                    "type": "rupture",
                    "date": day,
                    "product_id": int(r.product_id),
                    "product_name": r.name,
                    "title": f"Risque de rupture — {r.name}",
                    "description": (
                        f"Demande prévue {r.sales:.0f} unités (intervalle {r.lower:.0f}–{r.upper:.0f}), "
                        f"commande siège {r.seat_order}."
                    ),
                    "suggested_action": f"Commander {r.ai_order} unités (+{r.ai_order - r.seat_order}).",
                    "estimated_impact": round(float(impact), 2),
                })
        elif r.seat_order > r.upper:
            excess = r.seat_order - r.sales
            impact = excess * r.cost
            if impact >= MIN_IMPACT_EUR:
                alerts.append({
                    "id": f"gaspillage-{r.product_id}-{day}",
                    "type": "gaspillage",
                    "date": day,
                    "product_id": int(r.product_id),
                    "product_name": r.name,
                    "title": f"Risque de gaspillage — {r.name}",
                    "description": (
                        f"Commande siège {r.seat_order} pour une demande prévue de {r.sales:.0f} unités "
                        f"(intervalle {r.lower:.0f}–{r.upper:.0f}) : ~{excess:.0f} invendus."
                    ),
                    "suggested_action": f"Ramener la commande à {r.ai_order} unités.",
                    "estimated_impact": round(float(impact), 2),
                })

    for h in range(1, 8):
        day = store.reference_date + timedelta(days=h)
        for ev in store.events_on(day):
            alerts.append({
                "id": f"evenement-{ev['type']}-{day.isoformat()}",
                "type": "evenement",
                "date": day.isoformat(),
                "product_id": None,
                "product_name": None,
                "title": f"Événement : {ev['name']}",
                "description": f"Impact historique sur les ventes : x{ev['impact']:.1f} ({ev['type']}).",
                "suggested_action": "Vérifier les prévisions du jour et ajuster le personnel.",
                "estimated_impact": None,
                "severity_override": "high" if ev["impact"] >= 1.5 else "medium",
            })

    resolved = storage.resolved_alerts()
    for a in alerts:
        a["severity"] = a.pop("severity_override", None) or _severity(a["estimated_impact"] or 0)
        state = resolved.get(a["id"])
        a["resolved"] = state is not None
        a["resolved_at"] = state["resolved_at"] if state else None

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    alerts.sort(key=lambda a: (a["resolved"], order[a["severity"]], a["date"], -(a["estimated_impact"] or 0)))
    return alerts
