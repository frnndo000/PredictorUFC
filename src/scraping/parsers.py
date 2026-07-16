"""Parsers de las 4 páginas de ufcstats.com.

Cada función recibe un objeto BeautifulSoup y devuelve datos ya estructurados
(dicts / listas de dicts). No hacen peticiones de red: así se pueden testear
offline con los fixtures de `tests/fixtures/`.

Páginas:
  - parse_events_list(soup)  -> lista de eventos {event_id, name, date, location}
  - parse_event(soup)        -> lista de peleas del evento {fight_id, fighter1_id, fighter2_id}
  - parse_fight(soup)        -> detalle de una pelea (ganador, método, stats por peleador)
  - parse_fighter(soup)      -> ficha de un peleador (físico + stats de carrera)
"""
from __future__ import annotations

import re
from datetime import datetime

# --------------------------------------------------------------------------- #
# Helpers de limpieza / conversión
# --------------------------------------------------------------------------- #

def _id_from_url(url: str) -> str:
    """Último segmento de una URL de ufcstats = el id (evento/pelea/peleador)."""
    return url.rstrip("/").rsplit("/", 1)[-1]


def _clean(text: str | None) -> str:
    """Colapsa espacios en blanco y recorta."""
    return re.sub(r"\s+", " ", text).strip() if text else ""


def _parse_date(text: str):
    """'May 09, 2026' o 'Feb 27, 1991' -> datetime.date (ISO al escribir)."""
    text = _clean(text)
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _x_of_y(text: str):
    """'98 of 240' -> (98, 240). Celdas sin datos -> (None, None)."""
    m = re.search(r"(\d+)\s+of\s+(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _int(text: str):
    m = re.search(r"-?\d+", text or "")
    return int(m.group()) if m else None


def _pct(text: str):
    """'42%' -> 0.42 ; '---' -> None."""
    m = re.search(r"(\d+)\s*%", text or "")
    return int(m.group(1)) / 100 if m else None


def _float(text: str):
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(m.group()) if m else None


def _ctrl_to_seconds(text: str):
    """'7:16' -> 436 segundos ; '--' -> None."""
    m = re.search(r"(\d+):(\d+)", text or "")
    return int(m.group(1)) * 60 + int(m.group(2)) if m else None


def _height_to_inches(text: str):
    """'6\\' 1\"' -> 73 pulgadas."""
    m = re.search(r"(\d+)'\s*(\d+)", text or "")
    return int(m.group(1)) * 12 + int(m.group(2)) if m else None


# Orden importa: las variantes "Women's ..." y "Light Heavyweight" antes que las cortas.
_WEIGHT_CLASSES = [
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
    "Women's Featherweight", "Light Heavyweight", "Strawweight", "Flyweight",
    "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
    "Middleweight", "Heavyweight", "Catch Weight", "Open Weight",
]


def weight_class_from_bout(bout_type: str) -> str:
    """'UFC Middleweight Title Bout' -> 'Middleweight'. '' si no reconoce."""
    for wc in _WEIGHT_CLASSES:
        if wc.lower() in (bout_type or "").lower():
            return wc
    return ""


def normalize_method(raw: str) -> str:
    """Colapsa el método crudo de ufcstats a 4 clases: KO/TKO, SUB, DEC, OTHER."""
    r = (raw or "").lower()
    if "submission" in r:
        return "SUB"
    if "tko" in r or "ko" in r:
        return "KO/TKO"
    if "decision" in r:
        return "DEC"
    return "OTHER"


# --------------------------------------------------------------------------- #
# Parsers de página
# --------------------------------------------------------------------------- #

def parse_events_list(soup) -> list[dict]:
    """Eventos de la lista 'completed'. Salta el próximo evento (aún sin resultados)."""
    events = []
    for row in soup.select("tr.b-statistics__table-row"):
        link = row.select_one('a[href*="event-details"]')
        if link is None:
            continue
        # La fila del próximo evento trae un icono (next.png): la saltamos.
        if row.select_one("img.b-statistics__icon"):
            continue
        date = row.select_one("span.b-statistics__date")
        cols = row.select("td.b-statistics__table-col")
        location = _clean(cols[-1].get_text()) if cols else ""
        events.append({
            "event_id": _id_from_url(link["href"]),
            "name": _clean(link.get_text()),
            "date": _parse_date(date.get_text()) if date else None,
            "location": location,
        })
    return events


def parse_event(soup) -> list[dict]:
    """Peleas de un evento: id de la pelea y los dos peleadores (en orden)."""
    fights = []
    for row in soup.select("tr.js-fight-details-click"):
        fight_id = _id_from_url(row["data-link"])
        fighters = [_id_from_url(a["href"])
                    for a in row.select('a[href*="fighter-details"]')]
        if len(fighters) != 2:
            continue
        fights.append({
            "fight_id": fight_id,
            "fighter1_id": fighters[0],
            "fighter2_id": fighters[1],
        })
    return fights


def _fight_meta(soup) -> dict:
    """Método, round, tiempo, referee y tipo de combate (peso/título)."""
    meta = {}
    content = soup.select_one("div.b-fight-details__content")
    if content:
        for item in content.select("i.b-fight-details__label"):
            label = _clean(item.get_text()).rstrip(":").lower()
            # el valor es el texto del contenedor menos la etiqueta
            value = _clean(item.parent.get_text()).split(":", 1)[-1]
            if label in ("method", "round", "time", "referee"):
                meta[label] = _clean(value)
    title = soup.select_one("i.b-fight-details__fight-title")
    bout = _clean(title.get_text()) if title else ""
    meta["bout_type"] = bout
    meta["title_bout"] = "title" in bout.lower()
    return meta


def parse_fight(soup) -> dict:
    """Detalle de una pelea: ganador, método normalizado y stats por peleador."""
    persons = soup.select("div.b-fight-details__person")
    fighter_order, winner_id = [], None
    for p in persons:
        link = p.select_one("a.b-fight-details__person-link")
        status = _clean(p.select_one("i.b-fight-details__person-status").get_text())
        fid = _id_from_url(link["href"])
        fighter_order.append(fid)
        if status.upper() == "W":
            winner_id = fid

    meta = _fight_meta(soup)
    raw_method = meta.get("method", "")

    # Tabla "Totals" (primera tbody): una fila con dos <p> por columna.
    stats: dict[str, dict] = {}
    body = soup.select_one("tbody.b-fight-details__table-body")
    if body:
        cols = body.select("td.b-fight-details__table-col")
        fids = [_id_from_url(a["href"]) for a in cols[0].select('a[href*="fighter-details"]')]

        def pair(col):
            ps = cols[col].select("p.b-fight-details__table-text")
            return [_clean(p.get_text()) for p in ps]

        kd, sig, _sigpct, tot, td, _tdpct, sub, rev, ctrl = (pair(c) for c in range(1, 10))
        for i, fid in enumerate(fids):
            sig_l, sig_a = _x_of_y(sig[i])
            tot_l, tot_a = _x_of_y(tot[i])
            td_l, td_a = _x_of_y(td[i])
            stats[fid] = {
                "kd": _int(kd[i]),
                "sig_str_landed": sig_l, "sig_str_att": sig_a,
                "total_str_landed": tot_l, "total_str_att": tot_a,
                "td_landed": td_l, "td_att": td_a,
                "sub_att": _int(sub[i]), "rev": _int(rev[i]),
                "ctrl_sec": _ctrl_to_seconds(ctrl[i]),
            }

    return {
        "winner_id": winner_id,
        "fighter1_id": fighter_order[0] if fighter_order else None,
        "fighter2_id": fighter_order[1] if len(fighter_order) > 1 else None,
        "method_raw": raw_method,
        "method": normalize_method(raw_method),
        "round": _int(meta.get("round", "")),
        "time": meta.get("time", ""),
        "referee": meta.get("referee", ""),
        "bout_type": meta.get("bout_type", ""),
        "weight_class": weight_class_from_bout(meta.get("bout_type", "")),
        "title_bout": meta.get("title_bout", False),
        "stats": stats,
    }


def parse_fighter(soup) -> dict:
    """Ficha de un peleador: físico y estadísticas de carrera (valores actuales)."""
    name = soup.select_one("span.b-content__title-highlight")
    record = soup.select_one("span.b-content__title-record")
    wins = losses = draws = None
    if record:
        m = re.search(r"(\d+)-(\d+)-(\d+)", record.get_text())
        if m:
            wins, losses, draws = int(m.group(1)), int(m.group(2)), int(m.group(3))

    # Todos los items etiqueta:valor de las cajas (físico + carrera) en un dict.
    info = {}
    for li in soup.select("li.b-list__box-list-item_type_block"):
        title = li.find("i")
        if title is None:
            continue
        label = _clean(title.get_text()).rstrip(":")
        if not label:
            continue
        value = _clean(li.get_text()).split(":", 1)[-1]
        info[label] = _clean(value)

    return {
        "name": _clean(name.get_text()) if name else "",
        "wins": wins, "losses": losses, "draws": draws,
        "height_in": _height_to_inches(info.get("Height", "")),
        "weight_lbs": _int(info.get("Weight", "")),
        "reach_in": _int(info.get("Reach", "")),
        "stance": info.get("STANCE", "") or None,
        "dob": _parse_date(info.get("DOB", "")),
        "slpm": _float(info.get("SLpM", "")),
        "str_acc": _pct(info.get("Str. Acc.", "")),
        "sapm": _float(info.get("SApM", "")),
        "str_def": _pct(info.get("Str. Def", "")),  # ojo: sin punto final en el sitio
        "td_avg": _float(info.get("TD Avg.", "")),
        "td_acc": _pct(info.get("TD Acc.", "")),
        "td_def": _pct(info.get("TD Def.", "")),
        "sub_avg": _float(info.get("Sub. Avg.", "")),
    }
