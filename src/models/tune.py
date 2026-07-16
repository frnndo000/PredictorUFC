"""Búsqueda de hiperparámetros del modelo ganador (búsqueda aleatoria temporal).

Prueba muchas combinaciones entrenando en `train` y midiendo LogLoss en el
conjunto de **validación** (split temporal, no aleatorio, para no romper el
principio anti-leakage). El conjunto de **test** no se toca aquí: queda como
juez final honesto en `train.py`.

Guarda los mejores parámetros en models/best_params_winner.json, que `train.py`
carga automáticamente. Uso:  python -m src.models.tune  [--trials 40]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import log_loss

import config
from src.models.train import _features, _split_temporal

# Espacio de búsqueda (valores razonables por hiperparámetro).
SPACE = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63, 127],
    "max_depth": [-1, 4, 6, 8, 12],
    "min_child_samples": [20, 50, 100, 200],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
    "reg_alpha": [0.0, 0.5, 1.0, 2.0],
}
# Configuración por defecto actual (referencia a batir).
DEFAULT = {"learning_rate": 0.03, "num_leaves": 31, "max_depth": -1,
           "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
           "reg_lambda": 1.0, "reg_alpha": 0.0}


def _valid_logloss(params, Xtr, ytr, Xva, yva) -> float:
    """Entrena con early stopping en valid y devuelve su LogLoss."""
    model = LGBMClassifier(n_estimators=1500, subsample_freq=1,
                           random_state=config.RANDOM_SEED, verbose=-1, **params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[early_stopping(50, verbose=False)])
    return log_loss(yva, model.predict_proba(Xva)[:, 1])


def main(trials: int = 40) -> None:
    df = pd.read_csv(config.DATASET_CSV)
    train, valid, _ = _split_temporal(df)
    Xtr, ytr = _features(train), train["label"]
    Xva, yva = _features(valid), valid["label"]
    rng = np.random.default_rng(config.RANDOM_SEED)

    base = _valid_logloss(DEFAULT, Xtr, ytr, Xva, yva)
    print(f"config por defecto -> valid LogLoss {base:.4f}\n")
    best_ll, best_params = base, DEFAULT

    for i in range(1, trials + 1):
        params = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        ll = _valid_logloss(params, Xtr, ytr, Xva, yva)
        flag = ""
        if ll < best_ll:
            best_ll, best_params, flag = ll, params, "  <-- nuevo mejor"
        print(f"trial {i:2d}/{trials}: LogLoss {ll:.4f}"
              f"  (lr={params['learning_rate']}, leaves={params['num_leaves']}, "
              f"depth={params['max_depth']}, min_child={params['min_child_samples']}){flag}")

    improved = best_params is not DEFAULT
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BEST_PARAMS_WINNER, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nmejor valid LogLoss: {best_ll:.4f} "
          f"({'mejora' if improved else 'no mejora'} sobre {base:.4f})")
    print("parámetros guardados ->", config.BEST_PARAMS_WINNER)
    print(json.dumps(best_params, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    main(**vars(ap.parse_args()))
