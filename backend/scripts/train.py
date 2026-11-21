"""
Entraîne le LSTM multi-tâches et évalue le modèle sur la période de test.

    python -m scripts.train            (depuis le dossier backend/)

Produit :
    artifacts/model.pt       poids + configuration + features + normalisations
    artifacts/metrics.json   métriques de test (modèle vs baselines) et historique
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from app.config import settings
from app.data import DataStore
from app.ml.dataset import Samples, build_samples, compute_scales
from app.ml.features import LOOKBACK, RISK_LABELS, feature_names
from app.ml.model import MultiTaskLSTM

# Même découpage temporel que le module preprocessing
TRAIN_END = pd.Timestamp("2023-06-30")
VAL_END = pd.Timestamp("2023-09-30")
RISK_LOSS_WEIGHT = 0.3


def to_loader(s: Samples, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(s.X),
        torch.from_numpy(s.sales / s.scale),
        torch.from_numpy(s.waste / s.scale),
        torch.from_numpy(s.risk),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer=None) -> float:
    model.train(optimizer is not None)
    total, n = 0.0, 0
    for x, ys, yw, yr in loader:
        with torch.set_grad_enabled(optimizer is not None):
            ps, pw, logits = model(x)
            loss = F.smooth_l1_loss(ps, ys) + F.smooth_l1_loss(pw, yw) + RISK_LOSS_WEIGHT * F.cross_entropy(logits, yr)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


@torch.no_grad()
def predict(model, s: Samples, batch_size: int = 2048):
    model.eval()
    sales, waste, risk = [], [], []
    for i in range(0, len(s.X), batch_size):
        ps, pw, logits = model(torch.from_numpy(s.X[i:i + batch_size]))
        sales.append(ps.numpy())
        waste.append(pw.numpy())
        risk.append(logits.argmax(1).numpy())
    return np.concatenate(sales) * s.scale, np.concatenate(waste) * s.scale, np.concatenate(risk)


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {
        "mae": round(float(np.abs(err).mean()), 3),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        "wape": round(float(np.abs(err).sum() / max(y.sum(), 1e-9) * 100), 2),
        "bias": round(float(err.mean()), 3),
    }


def macro_f1(y: np.ndarray, p: np.ndarray, k: int = 3) -> float:
    f1s = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def order_simulation(s: Samples, order: np.ndarray, prices: np.ndarray, costs: np.ndarray) -> dict:
    """Invendus et ventes manquées si l'on avait commandé `order` chaque jour."""
    overstock = np.maximum(order - s.sales, 0)
    stockout = np.maximum(s.sales - order, 0)
    return {
        "units_ordered": int(order.sum()),
        "unsold_units": int(overstock.sum()),
        "unsold_cost_eur": round(float((overstock * costs).sum()), 2),
        "missed_sales_units": int(stockout.sum()),
        "missed_margin_eur": round(float((stockout * (prices - costs)).sum()), 2),
        "stockout_days_pct": round(float((stockout > 0).mean() * 100), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    store = DataStore.load(settings.DATA_DIR)
    categories = store.categories
    scales = compute_scales(store, TRAIN_END)
    samples = build_samples(store, categories, scales)
    dates = pd.DatetimeIndex(samples.dates)
    train = samples.subset(dates <= TRAIN_END)
    val = samples.subset((dates > TRAIN_END) & (dates <= VAL_END))
    test = samples.subset(dates > VAL_END)
    print(f"Séquences : train={len(train.X)}  val={len(val.X)}  test={len(test.X)}  features={samples.X.shape[2]}")

    model = MultiTaskLSTM(samples.X.shape[2], args.hidden_size, args.num_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    train_loader = to_loader(train, args.batch_size, shuffle=True)
    val_loader = to_loader(val, 2048, shuffle=False)

    best, best_state, wait, history = math.inf, None, 0, []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer)
        va = run_epoch(model, val_loader)
        scheduler.step(va)
        history.append({"epoch": epoch, "train_loss": round(tr, 5), "val_loss": round(va, 5)})
        print(f"epoch {epoch:02d}  train={tr:.4f}  val={va:.4f}")
        if va < best - 1e-4:
            best, wait = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping")
                break
    model.load_state_dict(best_state)

    # Intervalle de prédiction empirique (normalisé) mesuré sur la validation
    val_sales, _, _ = predict(model, val)
    rel = (val.sales - val_sales) / val.scale
    interval = {"q10": float(np.quantile(rel, 0.10)), "q90": float(np.quantile(rel, 0.90))}

    # ----------------------------------------------------------------- évaluation test
    p_sales, p_waste, p_risk = predict(model, test)
    prices = store.products.loc[test.product_ids, "price"].to_numpy()
    costs = store.products.loc[test.product_ids, "cost"].to_numpy()
    cats = store.products.loc[test.product_ids, "category"].to_numpy()

    in_interval = (test.sales >= p_sales + interval["q10"] * test.scale) & (test.sales <= p_sales + interval["q90"] * test.scale)
    majority = np.bincount(train.risk, minlength=3).argmax()

    ai_order = np.ceil(p_sales * (1 + settings.AI_SAFETY_MARGIN))
    seat_order = np.ceil(test.seat_mean4 * (1 + settings.SEAT_SAFETY_MARGIN))
    sim_ai = order_simulation(test, ai_order, prices, costs)
    sim_seat = order_simulation(test, seat_order, prices, costs)

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training_seconds": round(time.time() - t0, 1),
        "split": {
            "train": f"{store.first_date} → {TRAIN_END.date()}",
            "validation": f"{(TRAIN_END + pd.Timedelta(days=1)).date()} → {VAL_END.date()}",
            "test": f"{(VAL_END + pd.Timedelta(days=1)).date()} → {store.last_date}",
            "n_train": int(len(train.X)), "n_val": int(len(val.X)), "n_test": int(len(test.X)),
        },
        "horizon": "J+1 (prévision à un jour, historique réel)",
        "sales": {
            "lstm": regression_metrics(test.sales, p_sales),
            "naive_j7": regression_metrics(test.sales, test.lag7),
            "moyenne_4_semaines": regression_metrics(test.sales, test.seat_mean4),
        },
        "waste": {
            "lstm": regression_metrics(test.waste, p_waste),
            "naive_j7": regression_metrics(test.waste, test.lag7_waste),
        },
        "risk": {
            "labels": RISK_LABELS,
            "lstm_accuracy": round(float((p_risk == test.risk).mean()), 4),
            "lstm_macro_f1": round(macro_f1(test.risk, p_risk), 4),
            "majority_accuracy": round(float((test.risk == majority).mean()), 4),
        },
        "interval_80": {**{k: round(v, 4) for k, v in interval.items()},
                        "coverage_test_pct": round(float(in_interval.mean() * 100), 2)},
        "sales_wape_by_category": {
            c: {
                "lstm": regression_metrics(test.sales[cats == c], p_sales[cats == c])["wape"],
                "moyenne_4_semaines": regression_metrics(test.sales[cats == c], test.seat_mean4[cats == c])["wape"],
            }
            for c in categories
        },
        "order_simulation": {
            "description": (
                f"Commande IA = prévision x {1 + settings.AI_SAFETY_MARGIN:.2f} ; "
                f"commande siège = moyenne des 4 mêmes jours de semaine x {1 + settings.SEAT_SAFETY_MARGIN:.2f}"
            ),
            "ia": sim_ai,
            "siege": sim_seat,
        },
        "history": history,
    }

    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": model.config,
        "feature_names": feature_names(categories),
        "categories": categories,
        "lookback": LOOKBACK,
        "scales": scales,
        "interval": interval,
        "train_end": str(TRAIN_END.date()),
        "trained_at": metrics["trained_at"],
    }, settings.model_path)
    settings.metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    s = metrics["sales"]
    print(f"\nTest WAPE ventes : LSTM {s['lstm']['wape']}%  |  J-7 {s['naive_j7']['wape']}%  |  moy. 4 sem. {s['moyenne_4_semaines']['wape']}%")
    print(f"Couverture intervalle 80% : {metrics['interval_80']['coverage_test_pct']}%")
    print(f"Invendus (coût) IA {sim_ai['unsold_cost_eur']}€ vs siège {sim_seat['unsold_cost_eur']}€ ; "
          f"marge manquée IA {sim_ai['missed_margin_eur']}€ vs siège {sim_seat['missed_margin_eur']}€")
    print(f"Artefacts écrits dans {settings.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
