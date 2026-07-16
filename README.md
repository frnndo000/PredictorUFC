# 🥊 Predictor UFC

Pipeline de *machine learning* que predice el **ganador** y el **método de victoria**
(KO/TKO, Sumisión, Decisión) de una pelea de UFC, con una **simulación dinámica**:
eliges dos peleadores y el sistema devuelve la probabilidad de ganar de cada uno y la
distribución del método.

> ⚠️ Proyecto educativo / de portafolio. No está pensado para apuestas.

## Objetivo

Demostrar un flujo de trabajo completo y reproducible de data science aplicado al deporte:
obtención de datos propios, ingeniería de características rigurosa (sin fuga de datos),
modelado, evaluación honesta y una aplicación interactiva.

## Arquitectura del pipeline

```
ufcstats.com ──scraper──> data/raw/*.csv ──features──> data/processed/dataset.csv
                                                              │
                                                          modelos LightGBM
                                                              │
                                                     app Streamlit (simulación)
```

## Estructura del repositorio

| Ruta | Contenido |
|------|-----------|
| `src/scraping/` | Scraper incremental de ufcstats.com |
| `src/features/` | Ingeniería de características (leakage-safe) |
| `src/models/`   | Entrenamiento y predicción/simulación |
| `app/`          | Aplicación Streamlit |
| `notebooks/`    | EDA, features y evaluación de modelos |
| `tests/`        | Pruebas de parsers y guard anti-leakage |
| `data/`         | Datos crudos y procesados (se regeneran, no se versionan) |

## Reproducir (Windows / PowerShell)

```powershell
conda create -n predictor-ufc python=3.11 -y
conda activate predictor-ufc
pip install -r requirements.txt

python -m src.scraping.scraper        # 1) obtener datos
python -m src.features.build_features # 2) construir dataset
python -m src.models.train            # 3) entrenar modelos
streamlit run app/streamlit_app.py    # 4) simular peleas
```

## Estado

🚧 En construcción — se desarrolla por fases (ver historial de commits).
