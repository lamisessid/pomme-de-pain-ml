"""
Chargement du modèle entraîné et calcul des prévisions.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

from app.data import DataStore
from app.ml.dataset import CONTEXT_COLUMNS, build_samples
from app.ml.features import MAX_LAG, RISK_LABELS, build_frame, static_features
from app.ml.model import MultiTaskLSTM

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, model_path: Path, store: DataStore):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = MultiTaskLSTM(**ckpt["model_config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.categories: List[str] = ckpt["categories"]
        self.lookback: int = ckpt["lookback"]
        self.scales = {int(k): float(v) for k, v in ckpt["scales"].items()}
        self.interval = ckpt["interval"]
        self.trained_at: str = ckpt["trained_at"]
        self.store = store
        logger.info("Modèle chargé (%s features, entraîné le %s)", ckpt["model_config"]["input_size"], self.trained_at)

    # ------------------------------------------------------------------ prévision N jours
    def forecast(self, start: date, horizon: int, product_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
        """
        Prévoit les jours start .. start+horizon-1 à partir de l'historique réel jusqu'à start-1.
        Au-delà de J+1, les ventes prévues servent d'historique (prévision récursive).
        """
        store = self.store
        pids = sorted(product_ids) if product_ids else store.products.index.tolist()
        hist_len = self.lookback + MAX_LAG
        future = [start + timedelta(days=h) for h in range(horizon)]
        all_dates = pd.DatetimeIndex([start - timedelta(days=hist_len - i) for i in range(hist_len)] + future)
        context = pd.DataFrame([store.context_for(d.date()) for d in all_dates], index=all_dates)[CONTEXT_COLUMNS]

        sales = np.zeros((len(pids), len(all_dates)), dtype=np.float32)
        waste = np.zeros_like(sales)
        for i, pid in enumerate(pids):
            series = store.product_series(pid, end=start - timedelta(days=1)).reindex(all_dates[:hist_len])
            sales[i, :hist_len] = series["sales"].ffill().bfill().to_numpy()
            waste[i, :hist_len] = series["waste"].ffill().bfill().to_numpy()
        statics = [static_features(store.product(pid), self.categories) for pid in pids]
        scales = np.array([self.scales[pid] for pid in pids], dtype=np.float32)

        rows = []
        for h in range(horizon):
            t = hist_len + h
            X = np.stack([
                build_frame(sales[i, :t + 1], waste[i, :t + 1], all_dates[:t + 1], context.iloc[:t + 1],
                            scales[i], statics[i])[-self.lookback:]
                for i in range(len(pids))
            ])
            with torch.no_grad():
                ps, pw, logits = self.model(torch.from_numpy(X))
            ps, pw = ps.numpy() * scales, pw.numpy() * scales
            proba = torch.softmax(logits, dim=1).numpy()
            sales[:, t], waste[:, t] = ps, pw
            for i, pid in enumerate(pids):
                rows.append({
                    "date": all_dates[t].date(),
                    "product_id": pid,
                    "sales": float(ps[i]),
                    "waste": float(pw[i]),
                    "lower": float(max(ps[i] + self.interval["q10"] * scales[i], 0.0)),
                    "upper": float(ps[i] + self.interval["q90"] * scales[i]),
                    "risk": RISK_LABELS[int(proba[i].argmax())],
                    "risk_proba": {lbl: round(float(p), 3) for lbl, p in zip(RISK_LABELS, proba[i])},
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ backtest J+1
    @cached_property
    def backtest(self) -> pd.DataFrame:
        """Prévision J+1 (historique réel) pour chaque jour et produit de l'historique."""
        s = build_samples(self.store, self.categories, self.scales)
        preds_s, preds_w = [], []
        with torch.no_grad():
            for i in range(0, len(s.X), 4096):
                ps, pw, _ = self.model(torch.from_numpy(s.X[i:i + 4096]))
                preds_s.append(ps.numpy())
                preds_w.append(pw.numpy())
        return pd.DataFrame({
            "date": pd.to_datetime(s.dates),
            "product_id": s.product_ids,
            "actual_sales": s.sales,
            "actual_waste": s.waste,
            "pred_sales": np.concatenate(preds_s) * s.scale,
            "pred_waste": np.concatenate(preds_w) * s.scale,
            "seat_mean4": s.seat_mean4,
        })


_predictor: Optional[Predictor] = None


def load_predictor(model_path: Path, store: DataStore) -> Optional[Predictor]:
    global _predictor
    if not model_path.exists():
        logger.warning("Modèle introuvable (%s). Lancez `python -m scripts.train`.", model_path)
        _predictor = None
    else:
        _predictor = Predictor(model_path, store)
    return _predictor


def get_predictor() -> Optional[Predictor]:
    return _predictor
