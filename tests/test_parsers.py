"""Tests de los parsers contra HTML real guardado en tests/fixtures/.

Corren offline (sin red). Ejecutar: python -m pytest tests/test_parsers.py -v
"""
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

from src.scraping import parsers

FIXTURES = Path(__file__).parent / "fixtures"


def _soup(name):
    return BeautifulSoup((FIXTURES / name).read_text(encoding="utf-8"), "lxml")


def test_events_list():
    events = parsers.parse_events_list(_soup("events_list.html"))
    assert len(events) > 10
    # Ninguno debe ser el próximo evento (todos con fecha) y con id de 16 hex.
    assert all(e["event_id"] and e["date"] for e in events)
    ufc329 = next(e for e in events if "UFC 329" in e["name"])
    assert ufc329["date"] == date(2026, 7, 11)
    assert "Las Vegas" in ufc329["location"]


def test_parse_event():
    fights = parsers.parse_event(_soup("event.html"))
    assert len(fights) == 13  # UFC 328 tuvo 13 peleas
    main = fights[0]
    assert main["fight_id"] == "70f1621080b3435c"
    assert {main["fighter1_id"], main["fighter2_id"]} == {
        "0d8011111be000b2", "767755fd74662dbf"}


def test_parse_fight():
    f = parsers.parse_fight(_soup("fight.html"))
    assert f["winner_id"] == "0d8011111be000b2"      # Strickland ganó
    assert f["method"] == "DEC"
    assert f["method_raw"] == "Decision - Split"
    assert f["round"] == 5
    assert f["time"] == "5:00"
    assert f["referee"] == "Herb Dean"
    assert f["title_bout"] is True
    assert f["weight_class"] == "Middleweight"
    # Stats de Chimaev: 98 de 240 sig. strikes, 9 de 13 takedowns, 7:16 control.
    chimaev = f["stats"]["767755fd74662dbf"]
    assert chimaev["sig_str_landed"] == 98 and chimaev["sig_str_att"] == 240
    assert chimaev["td_landed"] == 9 and chimaev["td_att"] == 13
    assert chimaev["ctrl_sec"] == 7 * 60 + 16


def test_parse_fighter():
    ftr = parsers.parse_fighter(_soup("fighter.html"))
    assert ftr["name"] == "Sean Strickland"
    assert (ftr["wins"], ftr["losses"], ftr["draws"]) == (31, 7, 0)
    assert ftr["height_in"] == 73          # 6'1"
    assert ftr["reach_in"] == 76
    assert ftr["stance"] == "Orthodox"
    assert ftr["dob"] == date(1991, 2, 27)
    assert ftr["slpm"] == 5.98
    assert ftr["str_acc"] == 0.42
    assert ftr["str_def"] == 0.60          # el label 'Str. Def:' sin punto
    assert ftr["sub_avg"] == 0.1


def test_normalize_method():
    assert parsers.normalize_method("KO/TKO") == "KO/TKO"
    assert parsers.normalize_method("TKO - Doctor's Stoppage") == "KO/TKO"
    assert parsers.normalize_method("Submission") == "SUB"
    assert parsers.normalize_method("Decision - Unanimous") == "DEC"
    assert parsers.normalize_method("Could Not Continue") == "OTHER"
