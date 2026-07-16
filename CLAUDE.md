# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es

Pipeline de ML que predice el **ganador** y el **método de victoria** (KO/TKO, SUB, DEC) de una pelea de UFC, con una simulación dinámica (elegir 2 peleadores → probabilidades + método) entregada como app Streamlit. Datos propios scrapeados de `ufcstats.com`. Proyecto de portafolio; se construye por fases (ver README y commits).

## Entorno y comandos (importante)

El proyecto corre en un entorno conda **`predictor-ufc`** (Python 3.11), separado del `base` de Anaconda. Dos advertencias que ya causaron problemas:

- El código hace `import config` e `import src...` asumiendo la **raíz del proyecto en el path**. Al ejecutar fuera de la raíz falla con `ModuleNotFoundError`. Ejecutar siempre con `PYTHONPATH` apuntando a la raíz.
- En una terminal con el env activado, usar **`python -m pip`** (no `pip` a secas): el `pip` del `base` se cuela en el PATH e instala en el entorno equivocado.

```powershell
# Patrón robusto (independiente del directorio actual):
$root = "C:\Users\Fernando\Desktop\proyectos\predictor-ufc"; $env:PYTHONPATH = $root
$py = "C:\Users\Fernando\anaconda3\envs\predictor-ufc\python.exe"

& $py -m pytest -q                          # todos los tests (corren offline con fixtures)
& $py -m pytest tests/test_parsers.py::test_parse_fight -v   # un solo test
& $py -m src.scraping.scraper --limit 2     # scrape de prueba (2 eventos)
& $py -m src.scraping.scraper               # scrape completo (incremental, ~2-3 h la 1ra vez)
```

## Arquitectura

Flujo: `ufcstats.com → scraper → data/raw/*.csv → features → data/processed → modelos → app`.

### Scraping (`src/scraping/`, `src/utils.py`) — Fase 1, completa

Cuatro capas de páginas en ufcstats: lista de eventos → evento (peleas) → pelea (stats) → peleador (físico/carrera).

- **`utils.py::Fetcher`**: ufcstats protege las páginas con un reto anti-bot *proof-of-work* (devuelve "Checking your browser…" con un `nonce` y dificultad N; hay que encontrar `n` tal que `sha256(f"{nonce}:{n}")` empiece con N ceros, hacer POST a `/__c`, y el servidor da la cookie `_fmc`). `get_soup()` resuelve esto de forma transparente y reutiliza la cookie en la sesión. Si un parser deja de funcionar, primero revisar si el sitio cambió el reto.
- **`parsers.py`**: funciones **puras** (reciben `BeautifulSoup`, sin red) → testeables offline con `tests/fixtures/*.html`. Al tocar un parser, actualizar el fixture y el test correspondiente.
- **`scraper.py`**: orquestador **incremental e idempotente**. Lee los ids ya guardados en los CSV y solo procesa lo nuevo; marca cada evento como hecho al final (resumible ante interrupciones). `--limit N` toma los N eventos nuevos **más antiguos**.

Salida en `data/raw/` (gitignored): `events.csv`, `fights.csv`, `fight_stats.csv` (una fila por peleador-pelea), `fighters.csv`. Ids de ufcstats como claves.

### Config

`config.py` centraliza rutas, URLs, semilla y parámetros de scraping. No hardcodear rutas ni constantes en el resto del código.

## Principio rector: NO data leakage (crítico para features/modelo)

Las estadísticas *dentro* de una pelea (golpes, derribos, control) son **resultados**, no predictores — usarlas para predecir el ganador es fuga de datos. Igual, las stats de carrera de la ficha del peleador son **acumuladas actuales** (incluyen la propia pelea y las futuras). Las features deben reconstruirse **solo del historial previo** a la fecha de cada pelea. Este invariante debe cubrirse con un test (`tests/test_no_leakage.py`) cuando se construya la Fase 2.

## Estado por fases (todas completas)

1. ✅ Scraper (`src/scraping/`, `src/utils.py`) → `data/raw/*.csv`.
2. ✅ Features leakage-safe + temporales (`src/features/build_features.py`) → `data/processed/dataset.csv`. Acumulador cronológico (`_snapshot`/`_update`), simetría A/B. `latest_features()` reconstruye el estado actual de un peleador para simular.
3. ✅ Modelos LightGBM ganador + método (`src/models/train.py`), split **temporal**, `class_weight='balanced'` para el método. Artefacto en `models/models.joblib`. La calibración isotónica se evaluó pero no mejora → se envían probs crudas. `src/models/tune.py` hace búsqueda aleatoria temporal de hiperparámetros → `models/best_params_winner.json` (que `train.py` carga si existe).
4. ✅ Simulación (`src/models/predict.py::simulate_fight`, promedia ambas orientaciones por simetría) + app Streamlit (`app/streamlit_app.py`).

### Comandos del pipeline (con el patrón de entorno de arriba)

```powershell
& $py -m src.scraping.scraper            # actualizar datos (incremental)
& $py -m src.features.build_features     # reconstruir dataset
& $py -m src.models.tune                 # (opcional) afinar hiperparametros del ganador
& $py -m src.models.train                # reentrenar modelos
& $py -m streamlit run app/streamlit_app.py   # levantar la app
```
