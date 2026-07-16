"""Simulación de una pelea hipotética entre dos peleadores.

Toma el estado ACTUAL de cada peleador (features reconstruidas de todo su
historial), arma el vector de diferencias igual que en entrenamiento, y corre
los modelos promediando las dos orientaciones (A vs B y B vs A) para respetar
la simetría. Devuelve P(gana A), P(gana B) y la distribución del método.

Los datos y modelos se cargan una sola vez (cache de módulo).
"""
from __future__ import annotations

from functools import lru_cache

import joblib
import pandas as pd

import config
from src.features.build_features import ALL_FEATS, build_long, latest_features


@lru_cache(maxsize=1)
def _load():
    art = joblib.load(config.MODEL_PATH)
    fights = pd.read_csv(config.FIGHTS_CSV)
    stats = pd.read_csv(config.FIGHT_STATS_CSV)
    fighters = pd.read_csv(config.FIGHTERS_CSV).set_index("fighter_id")
    fighters["dob"] = pd.to_datetime(fighters["dob"], errors="coerce")
    now = pd.Timestamp.today()
    latest = latest_features(build_long(fights, stats), now)
    return art, fighters, latest, now


def list_fighters() -> pd.DataFrame:
    """Peleadores con al menos una pelea, para poblar los menús (id, nombre)."""
    _, fighters, latest, _ = _load()
    ids = [i for i in latest if i in fighters.index]
    df = fighters.loc[ids, ["name"]].reset_index()
    return df.sort_values("name").reset_index(drop=True)


def _static(fid, when, fighters):
    s = fighters.loc[fid] if fid in fighters.index else None
    age = (when - s["dob"]).days / 365.25 if s is not None and pd.notna(s["dob"]) else float("nan")
    return {"age": age,
            "height_in": s["height_in"] if s is not None else float("nan"),
            "reach_in": s["reach_in"] if s is not None else float("nan")}


def _row(fa, fb, stance_a, stance_b, weight_class, title_bout):
    row = {f"d_{k}": fa[k] - fb[k] for k in ALL_FEATS}
    row.update(title_bout=int(bool(title_bout)), weight_class=weight_class,
               stance_a=stance_a, stance_b=stance_b)
    return row


def simulate_fight(a_id: str, b_id: str, weight_class: str = "", title_bout: bool = False) -> dict:
    art, fighters, latest, now = _load()
    cols, cat = art["feature_cols"], art["cat_cols"]

    fa = {**latest[a_id], **_static(a_id, now, fighters)}
    fb = {**latest[b_id], **_static(b_id, now, fighters)}
    st_a = fighters.loc[a_id, "stance"] if a_id in fighters.index else None
    st_b = fighters.loc[b_id, "stance"] if b_id in fighters.index else None

    # Dos orientaciones para simetría.
    ab = _row(fa, fb, st_a, st_b, weight_class, title_bout)
    ba = _row(fb, fa, st_b, st_a, weight_class, title_bout)
    X = pd.DataFrame([ab, ba])[cols]
    for c in cat:
        X[c] = X[c].astype("category")

    # P(gana el primero): en 'ab' el primero es A; en 'ba' el primero es B.
    p_first = art["winner"].predict_proba(X)[:, 1]
    p_a = (p_first[0] + (1 - p_first[1])) / 2

    method_classes = list(art["method"].classes_)
    m = art["method"].predict_proba(X).mean(axis=0)  # promedio de ambas orientaciones
    methods = dict(zip(method_classes, m.round(4)))

    return {
        "name_a": fighters.loc[a_id, "name"] if a_id in fighters.index else a_id,
        "name_b": fighters.loc[b_id, "name"] if b_id in fighters.index else b_id,
        "p_a": round(float(p_a), 4), "p_b": round(float(1 - p_a), 4),
        "method": methods,
    }
