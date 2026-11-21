"""
API FastAPI — Pomme de Pain ML (prévision de la demande et optimisation des commandes, SQY).
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.data import get_store
from app.ml.predictor import get_predictor, load_predictor
from app.routes import alerts, chatbot, comparison, dashboard, events, forecasts, losses, products

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_predictor(settings.model_path, get_store())
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Prévision des ventes et des pertes par LSTM multi-tâches, recommandations de commande "
                "et alertes pour le point de vente Pomme de Pain de Saint-Quentin-en-Yvelines.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

API = "/api/v1"
app.include_router(dashboard.router, prefix=f"{API}/dashboard", tags=["Dashboard"])
app.include_router(forecasts.router, prefix=f"{API}/forecasts", tags=["Prévisions"])
app.include_router(losses.router, prefix=f"{API}/losses", tags=["Pertes"])
app.include_router(comparison.router, prefix=f"{API}/comparison", tags=["Siège vs IA"])
app.include_router(alerts.router, prefix=f"{API}/alerts", tags=["Alertes"])
app.include_router(products.router, prefix=f"{API}/products", tags=["Produits"])
app.include_router(events.router, prefix=f"{API}/events", tags=["Événements"])
app.include_router(chatbot.router, prefix=f"{API}/chatbot", tags=["Assistant"])


@app.get("/health", tags=["Système"])
def health():
    store = get_store()
    predictor = get_predictor()
    return {
        "status": "ok",
        "version": settings.VERSION,
        "model_loaded": predictor is not None,
        "model_trained_at": predictor.trained_at if predictor else None,
        "reference_date": store.reference_date.isoformat(),
        "data_range": {"start": store.first_date.isoformat(), "end": store.last_date.isoformat()},
        "products": int(len(store.products)),
    }
