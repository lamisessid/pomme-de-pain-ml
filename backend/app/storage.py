"""
Petite persistance SQLite : état (résolu / actif) des alertes générées.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Optional

from app.config import settings


@contextmanager
def _connect():
    settings.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alert_resolutions ("
            " alert_id TEXT PRIMARY KEY, resolved_at TEXT NOT NULL, note TEXT)"
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def resolved_alerts() -> Dict[str, dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT alert_id, resolved_at, note FROM alert_resolutions").fetchall()
    return {r[0]: {"resolved_at": r[1], "note": r[2]} for r in rows}


def resolve_alert(alert_id: str, note: Optional[str] = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO alert_resolutions (alert_id, resolved_at, note) VALUES (?, ?, ?)",
            (alert_id, datetime.now().isoformat(timespec="seconds"), note),
        )


def reopen_alert(alert_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM alert_resolutions WHERE alert_id = ?", (alert_id,))
