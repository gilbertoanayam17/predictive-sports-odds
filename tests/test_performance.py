"""Tests del análisis de rendimiento (habilidad frente a suerte)."""

from __future__ import annotations

import pandas as pd
import pytest

from predictive_odds.bet_tracker import BetTracker
from predictive_odds.domain import Selection
from predictive_odds.parlay_builder import Pick
from predictive_odds.performance import _streaks, breakdown, cumulative, summarize

KICK = pd.Timestamp("2026-11-08 15:00", tz="UTC")


def _closing(mid: str, home: float, draw: float, away: float) -> pd.DataFrame:
    base = {"match_id": mid, "league": "EPL", "home_team": "H", "away_team": "A", "commence_time": KICK,
            "bookmaker": "closing", "market": "1X2", "line": None}
    return pd.DataFrame([{**base, "selection": s, "odds": o} for s, o in (("HOME", home), ("DRAW", draw),
                                                                          ("AWAY", away))])


def _tracker() -> BetTracker:
    t = BetTracker(":memory:")
    for i, (league, won) in enumerate([("EPL", True), ("EPL", False), ("SERIE_A", False), ("SERIE_A", False)]):
        t.record_pick(Pick(f"m{i}", league, "H", "A", KICK, Selection.home(), 2.2, "b", 0.55), 10)
    t.update_closing(pd.concat([_closing(f"m{i}", 2.0, 3.6, 4.0) for i in range(3)]))  # m3 sin cierre
    t.settle(pd.DataFrame([{"match_id": f"m{i}", "home_team": "H", "away_team": "A", "date": KICK,
                            "home_goals": 2 if w else 0, "away_goals": 0 if w else 1, "status": "FT"}
                           for i, w in enumerate([True, False, False, False])]))
    return t


def test_summary_separates_skill_and_luck() -> None:
    with _tracker() as t:
        s = summarize(t.bets())
    assert s.bets == 4 and s.staked == 40
    assert s.profit == pytest.approx(12 - 30)
    assert s.closing_coverage == pytest.approx(0.75)
    # Esperado: 3 apuestas con p de cierre (~0.48) a 2.2 y 1 con p del modelo 0.55
    assert s.expected_profit > 0 and s.expected_from_closing > 0
    assert s.luck == pytest.approx(s.profit - s.expected_profit) and s.luck_z < 0
    assert s.longest_losing_streak == 3 and s.current_streak == -3
    assert "Muestra pequeña" in s.verdict()


def test_breakdown_and_cumulative() -> None:
    with _tracker() as t:
        bets = t.bets()
    by_league = breakdown(bets, "league").set_index("league")
    assert by_league.loc["EPL", "apuestas"] == 2 and by_league.loc["SERIE_A", "beneficio"] == -20
    by_market = breakdown(bets, "market")
    assert list(by_market["market"]) == ["1X2"]
    cum = cumulative(bets)
    assert len(cum) == 4 and cum["real"].iloc[-1] == pytest.approx(-18)


def test_empty_and_streaks() -> None:
    with BetTracker(":memory:") as t:
        assert summarize(t.bets()).bets == 0 and breakdown(t.bets(), "market").empty and cumulative(t.bets()).empty
    assert _streaks([True, True, False, True, True]) == (1, 2)
    assert _streaks([False, False, True, False]) == (2, -1)
