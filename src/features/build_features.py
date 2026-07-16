"""Construcción de features leakage-safe para el modelo.

Principio rector (ver CLAUDE.md): las features de una pelea deben calcularse
usando SOLO el historial previo a la fecha de esa pelea. Aquí se garantiza por
construcción: se recorren las peleas de cada peleador en orden cronológico y,
para cada una, se LEE el acumulador (estado previo del peleador) ANTES de
actualizarlo con el resultado de esa misma pelea.

Salida: data/processed/dataset.csv, con simetría A/B (cada pelea = 2 filas con
etiqueta invertida) y features como diferencias (peleador_A - peleador_B).

Uso:  python -m src.features.build_features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Features dinámicas por peleador (reconstruidas del historial previo).
DYNAMIC_FEATS = [
    "exp", "win_rate", "streak", "ko_rate", "sub_rate", "finish_rate",
    "days_since_last", "sig_pm", "sig_absorbed_pm", "sig_acc", "str_def",
    "td_pm", "td_acc", "td_def", "ctrl_pm", "sub_att_pm", "kd_pm", "avg_fight_time",
]
# Features estáticas (físico + edad a la fecha de la pelea).
STATIC_FEATS = ["age", "height_in", "reach_in"]
ALL_FEATS = DYNAMIC_FEATS + STATIC_FEATS


def _nz(x) -> float:
    """NaN -> 0.0 (para acumular sumas sin propagar NaN de peleas sin stats)."""
    return 0.0 if pd.isna(x) else float(x)


def _duration_sec(round_, time_str) -> int:
    """Duración total de la pelea: rounds completos * 5min + tiempo del último."""
    try:
        m, s = str(time_str).split(":")
        t = int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        t = 0
    r = int(round_) if not pd.isna(round_) else 1
    return max(0, r - 1) * 300 + t


def build_long(fights: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Tabla larga: una fila por (pelea, peleador) con sus stats y las del rival."""
    # Emparejar cada peleador con el otro de la misma pelea (self-join).
    merged = stats.merge(stats, on="fight_id", suffixes=("", "_opp"))
    merged = merged[merged["fighter_id"] != merged["fighter_id_opp"]]

    f = fights[["fight_id", "date", "winner_id", "method", "round", "time"]]
    long = merged.merge(f, on="fight_id")
    long["date"] = pd.to_datetime(long["date"])
    long["won"] = long["winner_id"] == long["fighter_id"]
    long["minutes"] = [
        _duration_sec(r, t) / 60 for r, t in zip(long["round"], long["time"])
    ]
    # Orden cronológico estable (dentro de un evento el orden exacto no importa).
    return long.sort_values(["date", "fight_id"]).reset_index(drop=True)


def compute_prefight_features(long: pd.DataFrame) -> pd.DataFrame:
    """Features PRE-pelea por (fight_id, fighter_id), leakage-safe por construcción."""
    acc: dict[str, dict] = {}
    rows = []

    for r in long.itertuples(index=False):
        a = acc.get(r.fighter_id)
        n = a["n"] if a else 0
        mins = a["minutes"] if a else 0.0

        def ratio(num, den):
            return (a[num] / a[den]) if a and a[den] else np.nan

        feat = {
            "fight_id": r.fight_id, "fighter_id": r.fighter_id,
            "exp": n,
            "win_rate": (a["wins"] / n) if n else np.nan,
            "streak": a["streak"] if a else 0,
            "ko_rate": (a["ko_wins"] / n) if n else np.nan,
            "sub_rate": (a["sub_wins"] / n) if n else np.nan,
            "finish_rate": ((a["ko_wins"] + a["sub_wins"]) / n) if n else np.nan,
            "days_since_last": (r.date - a["last_date"]).days if a else np.nan,
            "sig_pm": (a["sig_l"] / mins) if mins else np.nan,
            "sig_absorbed_pm": (a["osig_l"] / mins) if mins else np.nan,
            "sig_acc": ratio("sig_l", "sig_a"),
            "str_def": (1 - a["osig_l"] / a["osig_a"]) if a and a["osig_a"] else np.nan,
            "td_pm": (a["td_l"] / mins) if mins else np.nan,
            "td_acc": ratio("td_l", "td_a"),
            "td_def": (1 - a["otd_l"] / a["otd_a"]) if a and a["otd_a"] else np.nan,
            "ctrl_pm": (a["ctrl"] / mins) if mins else np.nan,
            "sub_att_pm": (a["sub"] / mins) if mins else np.nan,
            "kd_pm": (a["kd"] / mins) if mins else np.nan,
            "avg_fight_time": (mins / n) if n else np.nan,
        }
        rows.append(feat)

        # --- Actualizar el acumulador con ESTA pelea (después de leer) ---
        if a is None:
            a = acc[r.fighter_id] = {
                "n": 0, "wins": 0, "ko_wins": 0, "sub_wins": 0, "streak": 0,
                "minutes": 0.0, "sig_l": 0.0, "sig_a": 0.0, "osig_l": 0.0,
                "osig_a": 0.0, "td_l": 0.0, "td_a": 0.0, "otd_l": 0.0,
                "otd_a": 0.0, "ctrl": 0.0, "sub": 0.0, "kd": 0.0, "last_date": None,
            }
        a["n"] += 1
        if r.won:
            a["wins"] += 1
            a["streak"] = a["streak"] + 1 if a["streak"] > 0 else 1
            if r.method == "KO/TKO":
                a["ko_wins"] += 1
            elif r.method == "SUB":
                a["sub_wins"] += 1
        else:
            a["streak"] = a["streak"] - 1 if a["streak"] < 0 else -1
        a["minutes"] += r.minutes
        a["sig_l"] += _nz(r.sig_str_landed); a["sig_a"] += _nz(r.sig_str_att)
        a["osig_l"] += _nz(r.sig_str_landed_opp); a["osig_a"] += _nz(r.sig_str_att_opp)
        a["td_l"] += _nz(r.td_landed); a["td_a"] += _nz(r.td_att)
        a["otd_l"] += _nz(r.td_landed_opp); a["otd_a"] += _nz(r.td_att_opp)
        a["ctrl"] += _nz(r.ctrl_sec); a["sub"] += _nz(r.sub_att); a["kd"] += _nz(r.kd)
        a["last_date"] = r.date

    return pd.DataFrame(rows).set_index(["fight_id", "fighter_id"])


def assemble(fights, prefeat, fighters) -> pd.DataFrame:
    """Une features de ambos peleadores como diferencias + simetría A/B."""
    fighters = fighters.set_index("fighter_id")
    fighters["dob"] = pd.to_datetime(fighters["dob"], errors="coerce")

    def static(fid, when):
        s = fighters.loc[fid] if fid in fighters.index else None
        age = (when - s["dob"]).days / 365.25 if s is not None and pd.notna(s["dob"]) else np.nan
        return {
            "age": age,
            "height_in": s["height_in"] if s is not None else np.nan,
            "reach_in": s["reach_in"] if s is not None else np.nan,
        }

    rows = []
    for f in fights.itertuples(index=False):
        if pd.isna(f.winner_id):        # empates / no-contest: sin etiqueta
            continue
        a_id, b_id = f.fighter1_id, f.fighter2_id
        key_a, key_b = (f.fight_id, a_id), (f.fight_id, b_id)
        if key_a not in prefeat.index or key_b not in prefeat.index:
            continue
        when = pd.to_datetime(f.date)
        fa = {**prefeat.loc[key_a].to_dict(), **static(a_id, when)}
        fb = {**prefeat.loc[key_b].to_dict(), **static(b_id, when)}
        diff = {f"d_{k}": fa[k] - fb[k] for k in ALL_FEATS}

        base = {
            "fight_id": f.fight_id, "date": f.date,
            "title_bout": int(bool(f.title_bout)), "weight_class": f.weight_class,
            "method": f.method,
        }
        a_won = int(f.winner_id == a_id)
        stance_a = fighters.loc[a_id, "stance"] if a_id in fighters.index else None
        stance_b = fighters.loc[b_id, "stance"] if b_id in fighters.index else None

        # Fila original (A vs B)
        rows.append({**base, **diff, "stance_a": stance_a, "stance_b": stance_b,
                     "label": a_won})
        # Fila espejo (B vs A): diferencias negadas, etiqueta invertida
        rows.append({**base, **{k: -v for k, v in diff.items()},
                     "stance_a": stance_b, "stance_b": stance_a, "label": 1 - a_won})

    return pd.DataFrame(rows)


def main() -> None:
    fights = pd.read_csv(config.FIGHTS_CSV)
    stats = pd.read_csv(config.FIGHT_STATS_CSV)
    fighters = pd.read_csv(config.FIGHTERS_CSV)

    long = build_long(fights, stats)
    prefeat = compute_prefight_features(long)
    dataset = assemble(fights, prefeat, fighters)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(config.DATASET_CSV, index=False, encoding="utf-8")
    print(f"dataset: {len(dataset)} filas, {dataset.shape[1]} columnas -> {config.DATASET_CSV}")
    print("label balance:", dataset["label"].mean().round(3), "(debe ser ~0.5 por simetria)")
    print("metodo:\n" + dataset["method"].value_counts().to_string())


if __name__ == "__main__":
    main()
