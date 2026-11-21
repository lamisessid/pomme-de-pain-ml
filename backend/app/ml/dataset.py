"""
Construction des séquences (X, y) à partir du DataStore, partagée par l'entraînement et le backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from app.data import DataStore
from app.ml.features import LOOKBACK, MIN_TARGET_INDEX, build_frame, risk_class, static_features

CONTEXT_COLUMNS = ["temperature", "rain", "is_holiday", "has_event", "event_impact"]


@dataclass
class Samples:
    X: np.ndarray            # (N, LOOKBACK, F)
    sales: np.ndarray        # (N,) unités réelles
    waste: np.ndarray        # (N,)
    risk: np.ndarray         # (N,) classe 0/1/2
    scale: np.ndarray        # (N,) normalisation du produit
    dates: np.ndarray        # (N,) datetime64 du jour cible
    product_ids: np.ndarray  # (N,)
    seat_mean4: np.ndarray   # (N,) moyenne des 4 mêmes jours de semaine précédents (règle siège)
    lag7: np.ndarray         # (N,) ventes à J-7 (baseline naïve)
    lag7_waste: np.ndarray   # (N,)

    def subset(self, mask: np.ndarray) -> "Samples":
        return Samples(**{k: v[mask] for k, v in self.__dict__.items()})


def compute_scales(store: DataStore, until: pd.Timestamp) -> Dict[int, float]:
    df = store.sales[store.sales["date"] <= until]
    return {int(k): float(max(v, 1.0)) for k, v in df.groupby("product_id")["sales"].mean().items()}


def build_samples(store: DataStore, categories: Sequence[str], scales: Dict[int, float]) -> Samples:
    parts = {k: [] for k in Samples.__dataclass_fields__}
    for pid, df in store.sales.groupby("product_id"):
        df = df.sort_values("date")
        dates = pd.DatetimeIndex(df["date"])
        sales = df["sales"].to_numpy(dtype=np.float32)
        waste = df["waste"].to_numpy(dtype=np.float32)
        scale = scales[int(pid)]
        frame = build_frame(sales, waste, dates, df[CONTEXT_COLUMNS], scale,
                            static_features(store.product(int(pid)), categories))

        targets = np.arange(max(MIN_TARGET_INDEX, 28), len(df))
        win = np.lib.stride_tricks.sliding_window_view(frame, (LOOKBACK, frame.shape[1]))[:, 0]
        parts["X"].append(win[targets - LOOKBACK + 1])
        parts["sales"].append(sales[targets])
        parts["waste"].append(waste[targets])
        parts["risk"].append(risk_class(sales[targets], waste[targets]))
        parts["scale"].append(np.full(len(targets), scale, dtype=np.float32))
        parts["dates"].append(dates.values[targets])
        parts["product_ids"].append(np.full(len(targets), int(pid)))
        parts["seat_mean4"].append(np.mean([sales[targets - k] for k in (7, 14, 21, 28)], axis=0))
        parts["lag7"].append(sales[targets - 7])
        parts["lag7_waste"].append(waste[targets - 7])

    return Samples(**{k: np.concatenate(v) for k, v in parts.items()})
