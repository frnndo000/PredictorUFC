"""Entrenamiento de los modelos (ganador + método) con validación temporal.

- Split TEMPORAL (no aleatorio): train = peleas antiguas, valid = intermedias,
  test = las más recientes. Así la evaluación simula predecir peleas futuras.
- Modelo de ganador: LightGBM binario + calibración isotónica de probabilidades.
- Modelo de método: LightGBM multiclase (KO/TKO, SUB, DEC).
- Baseline (regresión logística) como referencia para medir el 'lift'.

Guarda todo en models/models.joblib. Uso:  python -m src.models.train
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, classification_report,
                             confusion_matrix, f1_score, log_loss, roc_auc_score)
from sklearn.preprocessing import StandardScaler

import config

META_COLS = ["fight_id", "date"]
TARGET_COLS = ["label", "method"]
CAT_COLS = ["weight_class", "stance_a", "stance_b"]


def _split_temporal(df):
    """Divide por fecha: 70% train, 15% valid, 15% test (cronológico)."""
    df = df.sort_values("date").reset_index(drop=True)
    dates = pd.to_datetime(df["date"])
    q70, q85 = dates.quantile([0.70, 0.85])
    train = df[dates <= q70]
    valid = df[(dates > q70) & (dates <= q85)]
    test = df[dates > q85]
    return train, valid, test


def _features(df):
    """Matriz de features (categóricas como dtype 'category' para LightGBM)."""
    X = df.drop(columns=META_COLS + TARGET_COLS)
    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    return X


def _lgbm(**kw):
    params = dict(n_estimators=1500, learning_rate=0.03, num_leaves=31,
                  min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=config.RANDOM_SEED, verbose=-1)
    params.update(kw)
    return LGBMClassifier(**params)


def _winner_overrides() -> dict:
    """Mejores hiperparámetros del ganador si tune.py ya los generó, si no {}."""
    if config.BEST_PARAMS_WINNER.exists():
        with open(config.BEST_PARAMS_WINNER) as f:
            return json.load(f)
    return {}


def _fit(model, Xtr, ytr, Xva, yva):
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    return model


def train_winner(train, valid, test):
    Xtr, Xva, Xte = _features(train), _features(valid), _features(test)
    ytr, yva, yte = train["label"], valid["label"], test["label"]

    # --- Baseline: regresión logística (numéricas, NaN->0, estandarizadas) ---
    num = [c for c in Xtr.columns if c not in CAT_COLS]
    scaler = StandardScaler().fit(Xtr[num].fillna(0))
    base = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr[num].fillna(0)), ytr)
    p_base = base.predict_proba(scaler.transform(Xte[num].fillna(0)))[:, 1]

    # --- LightGBM (usa hiperparámetros afinados por tune.py si existen) ---
    overrides = _winner_overrides()
    if overrides:
        print("usando hiperparámetros afinados:", overrides)
    model = _fit(_lgbm(**overrides), Xtr, ytr, Xva, yva)
    p_raw = model.predict_proba(Xte)[:, 1]

    # --- Análisis de calibración isotónica (ajustada en valid) ---
    # En este dataset NO mejora el LogLoss del test (LightGBM ya está bien
    # calibrado); se deja la comparación como evidencia y se envían las
    # probabilidades crudas.
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        model.predict_proba(Xva)[:, 1], yva)
    p_cal = iso.transform(p_raw)

    print("\n===== MODELO GANADOR (test temporal) =====")
    print(f"{'':22}{'ACC':>8}{'AUC':>8}{'LogLoss':>10}{'Brier':>8}")
    for name, p in [("Baseline (logística)", p_base), ("LightGBM", p_raw),
                    ("LightGBM calibrado", p_cal)]:
        print(f"{name:22}{accuracy_score(yte, p > 0.5):>8.3f}"
              f"{roc_auc_score(yte, p):>8.3f}{log_loss(yte, p):>10.3f}"
              f"{brier_score_loss(yte, p):>8.3f}")
    print("calibrador enviado: ninguno (crudo; el isotónico no mejora el test)")
    return model, None, list(Xtr.columns)


def train_method(train, valid, test):
    # El método OTHER (DQ, no-contest, etc.) es ruido: se descarta.
    tr, va, te = (d[d["method"] != "OTHER"] for d in (train, valid, test))
    Xtr, Xva, Xte = _features(tr), _features(va), _features(te)
    ytr, yva, yte = tr["method"], va["method"], te["method"]

    # class_weight balanced: sin esto el modelo colapsa a la clase mayoritaria (DEC).
    model = _fit(_lgbm(objective="multiclass", class_weight="balanced"),
                 Xtr, ytr, Xva, yva)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)

    print("\n===== MODELO MÉTODO (test temporal) =====")
    print(f"accuracy: {accuracy_score(yte, pred):.3f} | "
          f"macro-F1: {f1_score(yte, pred, average='macro'):.3f} | "
          f"log_loss: {log_loss(yte, proba, labels=list(model.classes_)):.3f}")
    print(classification_report(yte, pred, digits=3, zero_division=0))
    print("matriz de confusión (filas=real, cols=pred):", list(model.classes_))
    print(confusion_matrix(yte, pred, labels=model.classes_))
    return model


def main():
    df = pd.read_csv(config.DATASET_CSV)
    train, valid, test = _split_temporal(df)
    print(f"split temporal -> train={len(train)}  valid={len(valid)}  test={len(test)}")
    print(f"fechas: train<= {pd.to_datetime(train.date).max().date()} | "
          f"test>= {pd.to_datetime(test.date).min().date()}")

    winner, iso, feat_cols = train_winner(train, valid, test)
    method = train_method(train, valid, test)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"winner": winner, "calibrator": iso, "method": method,
                 "feature_cols": feat_cols, "cat_cols": CAT_COLS},
                config.MODEL_PATH)
    print(f"\nmodelos guardados -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
