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
import shap

import config
from src.features.build_features import ALL_FEATS, build_long, latest_features

# Nombres legibles de cada feature para la explicación SHAP.
FEATURE_LABELS = {
    "d_exp": "Experiencia (nº peleas)", "d_win_rate": "% de victorias",
    "d_streak": "Racha actual", "d_ko_rate": "Tasa de KO/TKO",
    "d_sub_rate": "Tasa de sumisiones", "d_finish_rate": "Tasa de finalización",
    "d_days_since_last": "Días desde última pelea", "d_sig_pm": "Golpes conectados/min",
    "d_sig_absorbed_pm": "Golpes recibidos/min", "d_sig_acc": "Precisión de golpeo",
    "d_str_def": "Defensa de golpeo", "d_td_pm": "Derribos/min",
    "d_td_acc": "Precisión de derribo", "d_td_def": "Defensa de derribo",
    "d_ctrl_pm": "Tiempo de control/min", "d_sub_att_pm": "Intentos de sumisión/min",
    "d_kd_pm": "Knockdowns/min", "d_avg_fight_time": "Duración media de peleas",
    "d_age": "Edad", "d_height_in": "Altura", "d_reach_in": "Alcance",
    "title_bout": "Pelea por título", "weight_class": "Categoría de peso",
    "stance_a": "Postura de A", "stance_b": "Postura de B",
}


@lru_cache(maxsize=1)
def _load():
    art = joblib.load(config.MODEL_PATH)
    fights = pd.read_csv(config.FIGHTS_CSV)
    stats = pd.read_csv(config.FIGHT_STATS_CSV)
    fighters = pd.read_csv(config.FIGHTERS_CSV).set_index("fighter_id")
    fighters["dob"] = pd.to_datetime(fighters["dob"], errors="coerce")
    now = pd.Timestamp.today()
    latest = latest_features(build_long(fights, stats), now)
    explainer = shap.TreeExplainer(art["winner"])
    return art, fighters, latest, now, explainer


def list_fighters() -> pd.DataFrame:
    """Peleadores con al menos una pelea, para poblar los menús (id, nombre)."""
    _, fighters, latest, _, _ = _load()
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


def _matchup_X(a_id, b_id, weight_class, title_bout):
    """DataFrame de 2 filas (orientación A-vs-B y B-vs-A) lista para el modelo."""
    art, fighters, latest, now, _ = _load()
    cols, cat = art["feature_cols"], art["cat_cols"]
    fa = {**latest[a_id], **_static(a_id, now, fighters)}
    fb = {**latest[b_id], **_static(b_id, now, fighters)}
    st_a = fighters.loc[a_id, "stance"] if a_id in fighters.index else None
    st_b = fighters.loc[b_id, "stance"] if b_id in fighters.index else None
    ab = _row(fa, fb, st_a, st_b, weight_class, title_bout)
    ba = _row(fb, fa, st_b, st_a, weight_class, title_bout)
    X = pd.DataFrame([ab, ba])[cols]
    for c in cat:
        X[c] = X[c].astype("category")
    return X


def simulate_fight(a_id: str, b_id: str, weight_class: str = "", title_bout: bool = False) -> dict:
    art, fighters, *_ = _load()
    X = _matchup_X(a_id, b_id, weight_class, title_bout)

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


def explain_winner(a_id: str, b_id: str, weight_class: str = "",
                   title_bout: bool = False, top: int = 8) -> list[dict]:
    """Contribuciones SHAP a P(gana A), orientación A-vs-B.

    Cada elemento: {'label', 'value'} donde value>0 favorece a A y value<0 a B.
    Devuelve las `top` features de mayor peso (por magnitud).
    """
    _, _, _, _, explainer = _load()
    X = _matchup_X(a_id, b_id, weight_class, title_bout).iloc[[0]]  # fila A-vs-B
    sv = explainer(X)
    vals = sv.values[0]
    if vals.ndim > 1:                    # clasificación binaria -> clase positiva (gana A)
        vals = vals[:, 1]
    contribs = [{"label": FEATURE_LABELS.get(c, c), "value": float(v)}
                for c, v in zip(X.columns, vals)]
    contribs.sort(key=lambda d: abs(d["value"]), reverse=True)
    return contribs[:top]
