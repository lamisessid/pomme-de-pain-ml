"""
Assistant opérationnel : détection d'intention par mots-clés et réponses construites à partir
des données réelles (prévisions, pertes, alertes) et de la base de procédures.
"""
import json
import unicodedata
from datetime import timedelta
from functools import lru_cache
from typing import List, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import BACKEND_DIR, settings
from app.data import get_store
from app.services.alerts import generate_alerts
from app.services.forecasting import upcoming

router = APIRouter()

KB_PATH = BACKEND_DIR / "app" / "knowledge" / "pomme_de_pain_kb.json"

SUGGESTIONS = [
    "Prévision des croissants pour demain",
    "Quelles sont les alertes actives ?",
    "Analyse du gaspillage ce mois-ci",
    "Quels produits se vendent le mieux ?",
    "Quels événements arrivent ?",
    "Quelle est la précision du modèle ?",
]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)


class ChatResponse(BaseModel):
    response: str
    intent: str
    suggestions: List[str]


JOURS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]


def jour(d) -> str:
    return f"{JOURS[d.weekday()]} {d:%d/%m}"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


@lru_cache(maxsize=1)
def knowledge() -> dict:
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def find_product(msg: str) -> Optional[Tuple[int, str]]:
    """Produit dont le nom (ou sa forme au pluriel) apparaît dans le message ; le plus long gagne."""
    best = None
    for p in get_store().products.itertuples():
        name = normalize(p.name)
        words = name.split()
        variants = {name, " ".join(w + "s" for w in words), words[0] + "s" if len(words) == 1 else name}
        if any(v in msg for v in variants) and (best is None or len(name) > len(best[1])):
            best = (int(p.id), p.name)
    return best


def procedure(proc_id: str) -> str:
    proc = next((p for p in knowledge()["procedures"] if p["id"] == proc_id), None)
    if not proc:
        return ""
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(proc["steps"], 1))
    return f"\n\n**Procédure — {proc['title']}**\n{steps}"


# ---------------------------------------------------------------------- intentions
def answer_forecast(msg: str) -> str:
    store = get_store()
    product = find_product(msg)
    df = upcoming(7)
    if product:
        pid, name = product
        rows = df[df["product_id"] == pid]
        lines = [
            f"- {jour(r.date)} : **{r.sales:.0f}** unités ({r.lower:.0f}–{r.upper:.0f}), "
            f"commande IA {r.ai_order} / siège {r.seat_order}"
            for r in rows.itertuples()
        ]
        return f"Prévisions pour **{name}** :\n" + "\n".join(lines)
    tomorrow = df[df["date"] == df["date"].min()].sort_values("sales", ascending=False)
    top = "\n".join(f"- {r.name} : {r.sales:.0f} unités (commande IA {r.ai_order})" for r in tomorrow.head(5).itertuples())
    ctx = store.context_for(tomorrow["date"].iloc[0])
    return (
        f"Demain ({jour(tomorrow['date'].iloc[0])}) : **{tomorrow['sales'].sum():.0f}** ventes prévues, "
        f"{tomorrow['waste'].sum():.0f} unités de pertes estimées. Météo {ctx['temperature']:.0f}°C.\n"
        f"Top 5 :\n{top}\n\nPrécisez un produit (ex. « prévision baguette ») pour le détail sur 7 jours."
    )


def answer_waste() -> str:
    store = get_store()
    end = store.reference_date
    df = store.sales_between(end - timedelta(days=29), end)
    sales, waste, cost = df["sales"].sum(), df["waste"].sum(), df["waste_cost"].sum()
    top = df.groupby("name")["waste_cost"].sum().sort_values(ascending=False).head(3)
    lines = "\n".join(f"- {n} : {c:.0f} €" for n, c in top.items())
    return (
        f"Sur les 30 derniers jours : **{waste:.0f}** unités perdues ({waste / sales * 100:.1f} % des ventes), "
        f"soit **{cost:.0f} €**.\nProduits les plus coûteux :\n{lines}" + procedure("PROC-001")
    )


def answer_alerts() -> str:
    alerts = [a for a in generate_alerts() if not a["resolved"]]
    if not alerts:
        return "Aucune alerte active."
    lines = "\n".join(f"- [{a['severity']}] {a['title']} ({a['date']}) — {a['suggested_action']}" for a in alerts[:5])
    return f"**{len(alerts)}** alertes actives. Les plus importantes :\n{lines}"


def answer_top() -> str:
    store = get_store()
    end = store.reference_date
    df = store.sales_between(end - timedelta(days=29), end)
    top = df.groupby("name").agg(sales=("sales", "sum"), revenue=("revenue", "sum")).sort_values("revenue", ascending=False).head(5)
    lines = "\n".join(f"- {n} : {r.sales:.0f} ventes, {r.revenue:.0f} €" for n, r in top.iterrows())
    return f"Meilleurs produits (chiffre d'affaires, 30 derniers jours) :\n{lines}"


def answer_events() -> str:
    store = get_store()
    ev = store.events[store.events["date"] > str(store.reference_date)].sort_values("date").head(5)
    if ev.empty:
        past = store.events.sort_values("date").tail(3)
        lines = "\n".join(f"- {r.date:%d/%m/%Y} : {r.name} (x{r.impact:.1f})" for r in past.itertuples())
        return f"Aucun événement connu après le {store.reference_date:%d/%m/%Y}. Derniers événements :\n{lines}" + procedure("PROC-003")
    lines = "\n".join(f"- {r.date:%d/%m/%Y} : {r.name} (impact x{r.impact:.1f})" for r in ev.itertuples())
    return f"Prochains événements SQY :\n{lines}" + procedure("PROC-003")


def answer_accuracy() -> str:
    if not settings.metrics_path.exists():
        return "Le modèle n'est pas encore entraîné."
    m = json.loads(settings.metrics_path.read_text())
    s = m["sales"]
    return (
        f"Sur la période de test ({m['split']['test']}), l'erreur relative (WAPE) du LSTM est de "
        f"**{s['lstm']['wape']} %**, contre {s['moyenne_4_semaines']['wape']} % pour la moyenne des 4 dernières semaines "
        f"et {s['naive_j7']['wape']} % pour la référence J-7. Erreur moyenne : {s['lstm']['mae']} unités par produit et par jour."
    )


def answer_faq(msg: str) -> Optional[str]:
    kb = knowledge()
    scored = []
    for item in kb["faq"] + kb["procedures"]:
        hits = sum(1 for k in item.get("keywords", []) if normalize(k) in msg)
        if hits:
            scored.append((hits, item))
    if not scored:
        return None
    item = max(scored, key=lambda x: x[0])[1]
    if "answer" in item:
        return item["answer"]
    return procedure(item["id"]).strip()


INTENTS = [
    ("accuracy", ("precision", "fiab", "modele", "erreur", "wape")),
    ("forecast", ("prevision", "prevoir", "demain", "prevu", "combien")),
    ("alerts", ("alerte",)),
    ("waste", ("gaspill", "perte", "invendu")),
    ("top", ("meilleur", "top", "vend le plus", "populaire")),
    ("events", ("evenement", "match", "rugby", "concert", "greve")),
]


@router.post("/message", response_model=ChatResponse)
def message(req: ChatRequest):
    msg = normalize(req.message)
    intent = next((name for name, keys in INTENTS if any(k in msg for k in keys)), None)
    if intent is None and find_product(msg):
        intent = "forecast"

    handlers = {
        "accuracy": answer_accuracy, "alerts": answer_alerts, "waste": answer_waste,
        "top": answer_top, "events": answer_events,
    }
    if intent == "forecast":
        response = answer_forecast(msg)
    elif intent in handlers:
        response = handlers[intent]()
    else:
        response = answer_faq(msg)
        intent = "faq" if response else "unknown"
        response = response or (
            "Je peux vous aider sur les prévisions, les alertes, le gaspillage, les meilleures ventes, "
            "les événements SQY et les procédures (rupture, invendus...). Essayez une des suggestions."
        )
    return ChatResponse(response=response, intent=intent, suggestions=SUGGESTIONS[:4])


@router.get("/suggestions")
def suggestions():
    return SUGGESTIONS
