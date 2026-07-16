"""Guard anti-leakage para la ingeniería de features.

Verifica que las features PRE-pelea se calculan usando SOLO el historial previo:
para un historial conocido a mano, comprueba experiencia, win_rate, racha y una
métrica por minuto acumulada. Si alguien introduce fuga de datos (usar la pelea
actual o futuras), estos asserts fallan.

Ejecutar: python -m pytest tests/test_no_leakage.py -v
"""
import numpy as np
import pandas as pd

from src.features.build_features import compute_prefight_features


def _long_row(fid, fighter, date, won, method, minutes, sig_l, sig_a):
    """Fila mínima de la tabla larga (rival con stats en 0 para simplificar)."""
    return {
        "fight_id": fid, "fighter_id": fighter, "date": pd.Timestamp(date),
        "won": won, "method": method, "minutes": minutes,
        "sig_str_landed": sig_l, "sig_str_att": sig_a,
        "sig_str_landed_opp": 0, "sig_str_att_opp": 0,
        "td_landed": 0, "td_att": 0, "td_landed_opp": 0, "td_att_opp": 0,
        "ctrl_sec": 0, "sub_att": 0, "kd": 0,
    }


def test_prefight_features_use_only_past():
    # Peleador X: 3 peleas cronológicas -> Gana, Pierde, Gana.
    long = pd.DataFrame([
        _long_row("f1", "X", "2020-01-01", True,  "KO/TKO", 5, 10, 20),
        _long_row("f2", "X", "2020-06-01", False, "DEC",    15, 30, 60),
        _long_row("f3", "X", "2021-01-01", True,  "SUB",    5, 5, 10),
    ])
    pf = compute_prefight_features(long)

    f1 = pf.loc[("f1", "X")]
    f2 = pf.loc[("f2", "X")]
    f3 = pf.loc[("f3", "X")]

    # 1ra pelea: sin historial previo.
    assert f1["exp"] == 0
    assert np.isnan(f1["win_rate"])
    assert f1["streak"] == 0
    assert np.isnan(f1["sig_pm"])

    # 2da pelea: 1 pelea previa (ganada por KO).
    assert f2["exp"] == 1
    assert f2["win_rate"] == 1.0
    assert f2["streak"] == 1
    assert f2["ko_rate"] == 1.0
    assert f2["sig_pm"] == 10 / 5            # solo la 1ra pelea
    assert f2["days_since_last"] == 152      # 2020-01-01 -> 2020-06-01

    # 3ra pelea: 2 previas (1-1), venía de perder -> racha -1.
    assert f3["exp"] == 2
    assert f3["win_rate"] == 0.5
    assert f3["streak"] == -1
    # sig_pm acumulado SOLO de las 2 previas, NUNCA de la actual:
    assert f3["sig_pm"] == (10 + 30) / (5 + 15)
    assert f3["sig_acc"] == (10 + 30) / (20 + 60)


def test_first_fight_of_everyone_is_blank():
    # Dos peleadores distintos, su primera aparición no debe tener historial.
    long = pd.DataFrame([
        _long_row("f1", "A", "2019-01-01", True,  "DEC", 15, 40, 80),
        _long_row("f1", "B", "2019-01-01", False, "DEC", 15, 20, 80),
    ])
    pf = compute_prefight_features(long)
    for fid in ("A", "B"):
        assert pf.loc[("f1", fid)]["exp"] == 0
        assert np.isnan(pf.loc[("f1", fid)]["win_rate"])
