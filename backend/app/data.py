"""
Accès aux données historiques (ventes, produits, météo, événements SQY).

Les CSV bruts sont chargés une seule fois en mémoire : le jeu fait ~36 500 lignes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import holidays
import numpy as np
import pandas as pd

from app.config import settings


@dataclass
class DataStore:
    sales: pd.DataFrame       # une ligne par (date, produit)
    products: pd.DataFrame    # indexé par id
    events: pd.DataFrame      # calendrier des événements SQY
    daily: pd.DataFrame       # contexte par jour (météo, férié, événement), indexé par date
    reference_date: date

    # ------------------------------------------------------------------ chargement
    @classmethod
    def load(cls, data_dir: Path, reference_date: Optional[str] = None) -> "DataStore":
        sales = pd.read_csv(data_dir / "ventes_journalieres.csv", parse_dates=["date"])
        sales = sales.rename(columns={
            "produit_id": "product_id",
            "ventes_unites": "sales",
            "pertes_unites": "waste",
            "pluie_mm": "rain",
            "est_ferie": "is_holiday",
            "a_evenement": "has_event",
            "types_evenements": "event_types",
            "impact_evenement": "event_impact",
            "chiffre_affaires": "revenue",
            "cout_pertes": "waste_cost",
        })
        sales["event_types"] = sales["event_types"].fillna("")
        sales = sales.sort_values(["product_id", "date"]).reset_index(drop=True)

        products = pd.read_csv(data_dir / "produits.csv").rename(columns={
            "nom": "code", "categorie": "category", "sous_categorie": "subcategory",
            "prix": "price", "cout": "cost", "duree_vie": "shelf_life_days",
            "popularite": "popularity", "marge": "margin",
        })
        products["name"] = products["code"].map(_pretty_name)
        products = products.set_index("id", drop=False).rename_axis(None)

        events = pd.read_csv(data_dir / "evenements_sqy.csv", parse_dates=["date"]).rename(columns={
            "nom_evenement": "name", "type_evenement": "type", "impact_ventes": "impact",
        })

        daily = (
            sales.groupby("date")[["temperature", "rain", "is_holiday", "has_event", "event_impact"]]
            .first()
            .sort_index()
        )

        last = sales["date"].max().date()
        ref = date.fromisoformat(reference_date) if reference_date else last - timedelta(days=14)
        return cls(sales=sales, products=products, events=events, daily=daily, reference_date=ref)

    # ------------------------------------------------------------------ helpers
    @property
    def first_date(self) -> date:
        return self.sales["date"].min().date()

    @property
    def last_date(self) -> date:
        return self.sales["date"].max().date()

    @property
    def categories(self) -> List[str]:
        return sorted(self.products["category"].unique().tolist())

    def product(self, product_id: int) -> dict:
        if product_id not in self.products.index:
            raise KeyError(product_id)
        return self.products.loc[product_id].to_dict()

    def sales_between(self, start: date, end: date) -> pd.DataFrame:
        """Ventes sur [start, end] (bornes incluses), enrichies des infos produit."""
        mask = (self.sales["date"] >= pd.Timestamp(start)) & (self.sales["date"] <= pd.Timestamp(end))
        df = self.sales.loc[mask]
        return df.merge(
            self.products[["id", "name", "category", "price", "cost"]],
            left_on="product_id", right_on="id", how="left",
        ).drop(columns="id")

    def product_series(self, product_id: int, end: Optional[date] = None) -> pd.DataFrame:
        df = self.sales[self.sales["product_id"] == product_id]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df.set_index("date")

    def context_for(self, day: date) -> Dict:
        """Contexte d'un jour. Hors historique : météo climatologique + jours fériés."""
        ts = pd.Timestamp(day)
        if ts in self.daily.index:
            row = self.daily.loc[ts]
            ctx = {
                "temperature": float(row["temperature"]),
                "rain": float(row["rain"]),
                "is_holiday": int(row["is_holiday"]),
                "has_event": int(row["has_event"]),
                "event_impact": float(row["event_impact"]),
                "source": "historique",
            }
        else:
            clim = self._climatology
            doy = min(ts.dayofyear, 365)
            ctx = {
                "temperature": float(clim.loc[doy, "temperature"]),
                "rain": float(clim.loc[doy, "rain"]),
                "is_holiday": int(day in holidays.France(years=day.year)),
                "has_event": 0,
                "event_impact": 1.0,
                "source": "climatologie",
            }
            ev = self.events[self.events["date"] == ts]
            if len(ev):
                ctx["has_event"] = 1
                ctx["event_impact"] = float(ev["impact"].max())
        ctx["events"] = self.events_on(day)
        return ctx

    def events_on(self, day: date) -> List[Dict]:
        ev = self.events[self.events["date"] == pd.Timestamp(day)]
        return [
            {"name": r["name"], "type": r["type"], "impact": float(r["impact"])}
            for _, r in ev.iterrows()
        ]

    @cached_property
    def _climatology(self) -> pd.DataFrame:
        d = self.daily[["temperature", "rain"]].copy()
        d["doy"] = d.index.dayofyear.clip(upper=365)
        clim = d.groupby("doy").mean()
        # lissage sur 15 jours pour éviter le bruit d'une seule année
        return clim.rolling(15, center=True, min_periods=1).mean()


DISPLAY_NAMES = {
    "flan_part": "Flan (part)", "donut_sucre": "Donut sucré", "muffin_xxl_coeur_noisette": "Muffin XXL cœur noisette",
    "pain_chocolat": "Pain au chocolat", "chausson_pommes": "Chausson aux pommes", "pain_raisin": "Pain aux raisins",
    "torsade_pepites_chocolat": "Torsade pépites chocolat", "salade_cesar": "Salade César",
    "croissant_jambon_comte": "Croissant jambon-comté", "cappuccino_m": "Cappuccino (M)", "cafe_glace": "Café glacé",
    "the_glace": "Thé glacé", "orange_presse_l": "Orange pressée (L)", "biere_33cl": "Bière 33 cl",
    "pain_graines": "Pain aux graines", "sandwich_poulet_caesar": "Sandwich poulet César",
    "sandwich_thon_crudites": "Sandwich thon crudités", "grilladin_burger_cheddar": "Grilladin burger cheddar",
    "hot_dog": "Hot-dog", "sandwich_comte": "Sandwich comté", "soupe_jour": "Soupe du jour",
    "croissant_comte": "Croissant comté", "formule_dej_sandwich": "Formule déj' sandwich",
    "formule_dej_salade": "Formule déj' salade", "panier_tgtg": "Panier Too Good To Go",
    "bouteille_eau": "Bouteille d'eau",
}


def _pretty_name(code: str) -> str:
    if code in DISPLAY_NAMES:
        return DISPLAY_NAMES[code]
    name = code.replace("_", " ").strip()
    return name[:1].upper() + name[1:]


_store: Optional[DataStore] = None


def get_store() -> DataStore:
    global _store
    if _store is None:
        _store = DataStore.load(settings.DATA_DIR, settings.REFERENCE_DATE)
    return _store


def daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège un sous-ensemble de ventes par jour."""
    out = df.groupby("date").agg(
        sales=("sales", "sum"), waste=("waste", "sum"),
        revenue=("revenue", "sum"), waste_cost=("waste_cost", "sum"),
    )
    out["waste_rate"] = np.where(out["sales"] > 0, out["waste"] / out["sales"], 0.0)
    return out
