"""Flujo real: liquidar apuestas cuyos IDs (Odds API) difieren de los de resultados (football-data)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from predictive_odds.bet_tracker import BetTracker
from predictive_odds.config import current_season
from predictive_odds.data_fetcher import DataFetcher, SimulatedDataProvider, SimulationConfig
from predictive_odds.domain import Selection
from predictive_odds.parlay_builder import Parlay, Pick
from predictive_odds.pipeline import AnalysisPipeline, PipelineConfig

KICK = pd.Timestamp("2026-11-08 15:00", tz="UTC")


def _pick(mid: str, home: str, away: str, sel: Selection, kickoff: pd.Timestamp | None = KICK) -> Pick:
    return Pick(mid, "EPL", home, away, kickoff, sel, 2.0, "b", 0.55)


def _results(rows: list[tuple[str, str, str, str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame([{"match_id": m, "home_team": h, "away_team": a, "date": pd.Timestamp(d, tz="UTC"),
                          "home_goals": hg, "away_goals": ag, "status": "FT"} for m, h, a, d, hg, ag in rows])


def test_settles_by_teams_and_kickoff_when_ids_differ() -> None:
    with BetTracker(":memory:") as t:
        t.record_pick(_pick("oa-abc123", "Man City", "Arsenal", Selection.home()), 10)
        t.record_parlay(Parlay((_pick("oa-1", "Man City", "Arsenal", Selection.over(2.5)),
                                _pick("oa-2", "Leeds", "Hull City", Selection.away())),
                               4.0, 0.3, 0.3, 0.25, 0.01), 5)
        results = _results([
            ("fd-EPL-1", "Man City", "Arsenal", "2026-11-08", 3, 1),
            ("fd-EPL-0", "Man City", "Arsenal", "2025-11-09", 0, 2),  # el partido de la temporada pasada no cuenta
        ])
        assert t.settle(results) == 1  # el parlay espera al Leeds-Hull
        assert t.settle(_results([("fd-EPL-1", "Man City", "Arsenal", "2026-11-08", 3, 1),
                                  ("fd-EPL-2", "Leeds", "Hull", "2026-11-09", 0, 1)])) == 1
        bets = t.bets().set_index("id")
        assert list(bets["status"]) == ["won", "won"]


def test_old_records_without_kickoff_use_creation_date() -> None:
    with BetTracker(":memory:") as t:
        t.record_pick(_pick("oa-x", "Fulham", "Everton", Selection.draw(), kickoff=None), 10)
        future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=3)
        old = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=200)
        assert t.settle(_results([("fd-0", "Fulham", "Everton", str(old.date()), 1, 1)])) == 0
        assert t.settle(_results([("fd-1", "Fulham", "Everton", str(future.date()), 1, 1)])) == 1


def test_current_season() -> None:
    assert current_season(date(2026, 11, 4)) == 2026
    assert current_season(date(2027, 3, 1)) == 2026
    assert current_season(date(2026, 7, 10)) == 2026  # arranca la Liga MX (Apertura)


def test_unknown_teams_are_reported() -> None:
    sim = SimulatedDataProvider(SimulationConfig(seed=1))
    ok = AnalysisPipeline(DataFetcher(sim, odds_provider=sim), PipelineConfig(use_ml=False)).run(evaluate=False)
    assert ok.unknown_teams == []

    class Promoted(SimulatedDataProvider):
        """Momios con un equipo recién ascendido que no está en el histórico."""

        def fetch_odds(self, league):  # type: ignore[no-untyped-def]
            odds = sim.fetch_odds(league)
            first = odds["home_team"].iloc[0]
            return odds.replace({first: "Coventry City"})

    promoted = Promoted(SimulationConfig(seed=1))
    r = AnalysisPipeline(DataFetcher(SimulatedDataProvider(SimulationConfig(seed=1, played_rounds_current=38)),
                                     odds_provider=promoted), PipelineConfig(use_ml=False)).run(evaluate=False)
    assert "Coventry City" in r.unknown_teams
