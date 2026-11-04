"""Tests de los componentes visuales (HTML) de la app."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predictive_odds import ui_html as ui
from predictive_odds.domain import MatchPrediction, Selection
from predictive_odds.parlay_builder import Parlay, Pick

KICK = pd.Timestamp("2026-11-08 15:30", tz="UTC")


def _pred(home: str = "Arsenal", away: str = "Chelsea") -> MatchPrediction:
    m = np.full((11, 11), 1 / 121)
    return MatchPrediction("m1", home, away, 1.6, 1.1, m, kickoff=KICK)


def test_prob_bar_writes_percentages_and_accessible_label() -> None:
    bar = ui.prob_bar(0.5, 0.3, 0.2)
    assert "50%" in bar and "30%" in bar and "20%" in bar
    assert 'role="img"' in bar and "Local 50%" in bar
    assert ui.HOME in bar and ui.DRAW in bar and ui.AWAY in bar
    tiny = ui.prob_bar(0.9, 0.06, 0.04)
    assert ">6%<" not in tiny  # segmentos estrechos sin texto (no se solapa); el % sigue en el title


def test_spanish_dates() -> None:
    assert ui.fecha(KICK) == "dom 08 nov · 15:30"
    assert ui.fecha(None) == "" and ui.fecha(pd.NaT) == "" and ui.fecha("no-fecha") == ""


def test_cards_escape_dynamic_text() -> None:
    evil = "<script>alert(1)</script>"
    card = ui.match_card(_pred(home=evil), value_count=2)
    assert "<script>" not in card and "&lt;script&gt;" in card and "2 valor" in card
    pick = Pick("m1", "EPL", evil, "B", KICK, Selection.over(2.5), 2.1, "<b>book</b>", 0.55, 0.5, 0.02,
                decision_prob=0.53)
    ticket = ui.pick_ticket(pick, 1000)
    assert "<script>" not in ticket and "&lt;b&gt;book" in ticket and "2.10" in ticket and "20.00" in ticket
    par = Parlay((pick, pick), 4.41, 0.3, 0.3, 0.25, 0.01)
    assert "<script>" not in ui.parlay_ticket(par, 1000) and "4.41" in ui.parlay_ticket(par, 1000)
    assert "&lt;" in ui.empty("<x>")


def test_masthead_and_counts() -> None:
    head = ui.masthead("Premier <League>", live=True, season=2026)
    assert "Datos reales" in head and "Temporada 2026/27" in head and "Premier &lt;League&gt;" in head
    assert "Simulación" in ui.masthead("EPL", live=False)
    picks = [Pick(m, "EPL", "H", "A", None, Selection.home(), 2.0, "b", 0.6) for m in ("a", "a", "b")]
    assert ui.value_counts(picks) == {"a": 2, "b": 1}
    assert ui.grid(["<i>x</i>"]).startswith('<div class="grid">')
