"""Utilidades de red para el scraper.

ufcstats.com protege sus páginas con un reto anti-bot de tipo *proof-of-work*:
la primera respuesta es una página "Checking your browser…" que trae un `nonce`
y una dificultad `N`. Hay que encontrar un número `n` tal que
`sha256(f"{nonce}:{n}")` empiece con `N` ceros hexadecimales, enviarlo por POST
a `/__c`, y el servidor entrega una cookie de acceso (`_fmc`). Esa cookie se
reutiliza en toda la sesión, así que el reto solo se resuelve una vez (y se
vuelve a resolver solo si la cookie expira).

`Fetcher.get_soup(url)` encapsula todo esto: devuelve un BeautifulSoup del HTML
real, resolviendo el reto de forma transparente cuando aparece.
"""
from __future__ import annotations

import hashlib
import re
import time

import requests
from bs4 import BeautifulSoup

import config

CHALLENGE_MARKER = "Checking your browser"
_NONCE_RE = re.compile(r'nonce="([0-9a-f]+)"')
_ZEROS_RE = re.compile(r'new Array\((\d+)\+1\)')


class Fetcher:
    """Sesión HTTP que resuelve el reto proof-of-work de ufcstats.com."""

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.USER_AGENT})

    def _solve_challenge(self, html: str) -> None:
        """Resuelve el proof-of-work y obtiene la cookie de acceso."""
        nonce = _NONCE_RE.search(html).group(1)
        zeros = int(_ZEROS_RE.search(html).group(1))
        prefix = "0" * zeros
        n = 0
        while not hashlib.sha256(f"{nonce}:{n}".encode()).hexdigest().startswith(prefix):
            n += 1
        self.session.post(
            f"{config.BASE_URL}/__c",
            data={"nonce": nonce, "n": n},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=config.REQUEST_TIMEOUT_SEC,
        )

    def get_soup(self, url: str) -> BeautifulSoup:
        """Descarga `url` y devuelve el HTML parseado, sorteando el reto anti-bot."""
        last_error: Exception | None = None
        for _ in range(config.MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT_SEC)
            except requests.RequestException as exc:  # red caída, timeout, etc.
                last_error = exc
                time.sleep(config.REQUEST_DELAY_SEC * 2)
                continue
            if CHALLENGE_MARKER in resp.text:
                self._solve_challenge(resp.text)  # y reintenta con la cookie ya puesta
                continue
            resp.raise_for_status()
            time.sleep(config.REQUEST_DELAY_SEC)  # pausa cortés entre requests
            return BeautifulSoup(resp.text, "lxml")
        raise RuntimeError(f"No se pudo obtener {url} tras {config.MAX_RETRIES} intentos ({last_error})")
