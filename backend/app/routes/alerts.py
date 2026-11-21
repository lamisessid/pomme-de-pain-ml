"""
Alertes générées à partir des prévisions ; leur résolution est persistée en SQLite.
"""
from collections import Counter
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import storage
from app.services.alerts import generate_alerts

router = APIRouter()


class ResolveBody(BaseModel):
    note: Optional[str] = None


@router.get("")
def list_alerts(status: Literal["active", "resolved", "all"] = "active", type: Optional[str] = None):
    alerts = generate_alerts()
    if status != "all":
        alerts = [a for a in alerts if a["resolved"] == (status == "resolved")]
    if type:
        alerts = [a for a in alerts if a["type"] == type]
    return alerts


@router.get("/stats")
def stats():
    alerts = generate_alerts()
    active = [a for a in alerts if not a["resolved"]]
    return {
        "total": len(alerts),
        "active": len(active),
        "resolved": len(alerts) - len(active),
        "by_type": dict(Counter(a["type"] for a in active)),
        "by_severity": dict(Counter(a["severity"] for a in active)),
        "estimated_impact": round(sum(a["estimated_impact"] or 0 for a in active), 2),
    }


def _ensure_exists(alert_id: str) -> None:
    if alert_id not in {a["id"] for a in generate_alerts()}:
        raise HTTPException(404, "Alerte introuvable")


@router.post("/{alert_id}/resolve")
def resolve(alert_id: str, body: ResolveBody = ResolveBody()):
    _ensure_exists(alert_id)
    storage.resolve_alert(alert_id, body.note)
    return {"id": alert_id, "resolved": True}


@router.post("/{alert_id}/reopen")
def reopen(alert_id: str):
    _ensure_exists(alert_id)
    storage.reopen_alert(alert_id)
    return {"id": alert_id, "resolved": False}
