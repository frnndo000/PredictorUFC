"""Scraper incremental de ufcstats.com.

Recorre la lista de eventos completados y, para cada evento que aún NO está en
`data/raw/events.csv`, descarga sus peleas y las fichas de los peleadores nuevos,
agregando todo a los CSV crudos. Es idempotente (por id de evento/pelea/peleador)
y resumible: si se interrumpe, al re-ejecutar continúa donde quedó.

Uso:
    python -m src.scraping.scraper            # todos los eventos nuevos
    python -m src.scraping.scraper --limit 2  # solo 2 eventos nuevos (para probar)
"""
from __future__ import annotations

import argparse

import pandas as pd

import config
from src.scraping import parsers
from src.utils import Fetcher

# Columnas (y su orden) de cada CSV crudo.
EVENT_COLS = ["event_id", "name", "date", "location"]
FIGHT_COLS = ["fight_id", "event_id", "date", "winner_id", "fighter1_id", "fighter2_id",
              "method", "method_raw", "round", "time", "referee",
              "weight_class", "title_bout", "bout_type"]
STAT_COLS = ["fight_id", "fighter_id", "kd", "sig_str_landed", "sig_str_att",
             "total_str_landed", "total_str_att", "td_landed", "td_att",
             "sub_att", "rev", "ctrl_sec"]
FIGHTER_COLS = ["fighter_id", "name", "wins", "losses", "draws", "height_in",
                "weight_lbs", "reach_in", "stance", "dob", "slpm", "str_acc",
                "sapm", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]


def _known_ids(path, col) -> set:
    """Ids ya guardados en un CSV (vacío si el archivo no existe todavía)."""
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=[col])[col].astype(str))


def _append(path, rows, columns) -> None:
    """Agrega filas a un CSV, escribiendo la cabecera solo si el archivo es nuevo."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def scrape(limit: int | None = None) -> None:
    fetcher = Fetcher()

    known_events = _known_ids(config.EVENTS_CSV, "event_id")
    known_fights = _known_ids(config.FIGHTS_CSV, "fight_id")
    known_fighters = _known_ids(config.FIGHTERS_CSV, "fighter_id")

    events = parsers.parse_events_list(fetcher.get_soup(config.EVENTS_LIST_URL))
    new_events = [e for e in events if e["event_id"] not in known_events]
    new_events.reverse()  # procesar de más antiguo a más nuevo (dataset cronológico)
    if limit:
        new_events = new_events[:limit]

    print(f"Eventos en el sitio: {len(events)} | nuevos por procesar: {len(new_events)}")

    for i, ev in enumerate(new_events, 1):
        print(f"[{i}/{len(new_events)}] {ev['name']} ({ev['date']})")
        event_url = f"{config.BASE_URL}/event-details/{ev['event_id']}"
        fights = parsers.parse_event(fetcher.get_soup(event_url))

        for fr in fights:
            if fr["fight_id"] in known_fights:
                continue
            fight_url = f"{config.BASE_URL}/fight-details/{fr['fight_id']}"
            fd = parsers.parse_fight(fetcher.get_soup(fight_url))

            _append(config.FIGHTS_CSV, [{
                "fight_id": fr["fight_id"], "event_id": ev["event_id"], "date": ev["date"],
                "winner_id": fd["winner_id"],
                "fighter1_id": fd["fighter1_id"] or fr["fighter1_id"],
                "fighter2_id": fd["fighter2_id"] or fr["fighter2_id"],
                "method": fd["method"], "method_raw": fd["method_raw"],
                "round": fd["round"], "time": fd["time"], "referee": fd["referee"],
                "weight_class": fd["weight_class"], "title_bout": fd["title_bout"],
                "bout_type": fd["bout_type"],
            }], FIGHT_COLS)

            _append(config.FIGHT_STATS_CSV, [
                {"fight_id": fr["fight_id"], "fighter_id": fid, **s}
                for fid, s in fd["stats"].items()
            ], STAT_COLS)
            known_fights.add(fr["fight_id"])

            # Fichas de peleadores nuevos (snapshot único: el físico no cambia).
            # ponytail: no re-scrapeamos fichas ya vistas; las stats de carrera se
            # reconstruyen del historial en la fase de features, no de aquí.
            for fid in (fr["fighter1_id"], fr["fighter2_id"]):
                if fid in known_fighters:
                    continue
                ftr = parsers.parse_fighter(
                    fetcher.get_soup(f"{config.BASE_URL}/fighter-details/{fid}"))
                _append(config.FIGHTERS_CSV, [{"fighter_id": fid, **ftr}], FIGHTER_COLS)
                known_fighters.add(fid)

        # El evento se marca como hecho al final => resumible sin duplicar.
        _append(config.EVENTS_CSV, [ev], EVENT_COLS)

    print("Listo.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Scraper incremental de ufcstats.com")
    ap.add_argument("--limit", type=int, default=None,
                    help="procesar solo N eventos nuevos (para pruebas)")
    scrape(**vars(ap.parse_args()))


if __name__ == "__main__":
    main()
