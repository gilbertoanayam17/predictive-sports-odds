"""Tests del registro de apuestas (SQLite)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from predictive_odds.bet_tracker import BetTracker, TrackerError, parse_selection_code, selection_code
from predictive_odds.config import get_league
from predictive_odds.data_fetcher import DataFetcher, SimulatedDataProvider, SimulationConfig
from predictive_odds.domain import Selection
from predictive_odds.parlay_builder import Parlay, Pick
from predictive_odds.pipeline import AnalysisPipeline, PipelineConfig


def _pick(mid: str, sel: Selection, odds: float = 2.0) -> Pick:
    return Pick(mid, "EPL", "H", "A", None, sel, odds, "book", 0.55, 0.5, 0.02)


def _results(rows: list[tuple[str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame([{"match_id": m, "home_goals": h, "away_goals": a, "status": "FT"} for m, h, a in rows])


def test_selection_code_roundtrip() -> None:
    for sel in (Selection.home(), Selection.over(2.5), Selection.btts(False)):
        assert parse_selection_code(selection_code(sel)) == sel
    with pytest.raises(ValueError):
        parse_selection_code("1X2|HOME")


def test_record_settle_and_summary(tmp_path: Path) -> None:
    with BetTracker(tmp_path / "bets.sqlite") as t:
        assert t.record_pick(_pick("m1", Selection.home()), 10) is not None
        assert t.record_pick(_pick("m1", Selection.home()), 10) is None  # idempotente
        t.record_pick(_pick("m2", Selection.over(2.5), 1.9), 20)
        legs = (_pick("m1", Selection.home()), _pick("m3", Selection.btts(True)))
        t.record_parlay(Parlay(legs, 4.0, 0.3, 0.3, 0.25, 0.01), 5)
        with pytest.raises(ValueError):
            t.record_pick(_pick("m9", Selection.home()), 0)

        assert t.settle(_results([("m1", 2, 0), ("m2", 1, 1)])) == 2  # el parlay espera a m3
        s = t.summary(initial_bankroll=1000)
        assert s["beneficio"] == pytest.approx(10 * 1.0 - 20)
        assert s["pendientes"] == 1 and s["exposicion_abierta"] == 5
        assert t.settle(_results([("m1", 2, 0), ("m3", 1, 1)])) == 1
        assert t.summary()["beneficio"] == pytest.approx(-10 + 5 * 3.0)
        assert set(t.by_market()["grupo"]) == {"1X2", "OU", "PARLAY"}
        assert t.bankroll_history(1000)["bankroll"].iloc[-1] == pytest.approx(1005)

    # Persistencia: reabrir conserva los datos
    with BetTracker(tmp_path / "bets.sqlite") as t2:
        assert len(t2.bets()) == 3
        t2.void(1)
        assert t2.bets().set_index("id").loc[1, "status"] == "void"
        with pytest.raises(TrackerError):
            t2.void(999)


def test_closing_and_clv() -> None:
    with BetTracker(":memory:") as t:
        t.record_pick(_pick("m1", Selection.home(), 2.2), 10)
        closing = pd.DataFrame([
            {"match_id": "m1", "league": "EPL", "home_team": "H", "away_team": "A", "commence_time": None,
             "bookmaker": "closing", "market": "1X2", "selection": s, "line": None, "odds": o}
            for s, o in (("HOME", 2.0), ("DRAW", 3.6), ("AWAY", 4.0))])
        assert t.update_closing(closing) == 1
        assert t.bets()["clv"].iloc[0] > 0  # apostamos a 2.2 y cerró a ~2.0


def test_end_to_end_with_simulation(tmp_path: Path) -> None:
    sim = SimulatedDataProvider(SimulationConfig(seed=4, played_rounds_current=28))
    result = AnalysisPipeline(DataFetcher(sim, odds_provider=sim), PipelineConfig(use_ml=False)).run(evaluate=False)
    with BetTracker(tmp_path / "e2e.sqlite") as t:
        n = t.record_analysis(result.value_bets, result.parlays, result.bankroll)
        assert n > 0
        # Se juega la jornada 29 (mismos resultados previos gracias a la semilla)
        later = SimulatedDataProvider(SimulationConfig(seed=4, played_rounds_current=29))
        settled = t.settle(later.fetch_matches(get_league("EPL"), 2025))
        assert settled == n
        assert t.summary()["pendientes"] == 0
