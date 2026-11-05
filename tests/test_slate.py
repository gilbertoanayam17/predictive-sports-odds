"""Tests de la jornada multiliga."""

from __future__ import annotations

import pytest

from predictive_odds.data_fetcher import DataFetcher, DataFetchError, SimulatedDataProvider
from predictive_odds.pipeline import PipelineConfig, analyze_slate


def test_slate_combines_leagues_with_global_cap() -> None:
    sim = SimulatedDataProvider()
    cfg = PipelineConfig(use_ml=False, max_singles=15, max_total_exposure=0.08)
    slate = analyze_slate(DataFetcher(sim, odds_provider=sim), ["EPL", "LIGA_MX", "UCL"], cfg)
    assert set(slate.results) == {"EPL", "LIGA_MX", "UCL"} and not slate.errors
    assert len(slate.value_bets) == 15
    assert slate.total_stake <= 0.08 * slate.bankroll + 1e-9
    leagues_in_parlays = {leg.league for p in slate.parlays for leg in p.legs}
    assert len(leagues_in_parlays) >= 2
    assert set(slate.value_bets_frame()["liga"]) <= {"EPL", "LIGA_MX", "UCL"}


def test_slate_tolerates_failing_league() -> None:
    sim = SimulatedDataProvider()

    class Flaky(SimulatedDataProvider):
        def fetch_matches(self, league, season):  # type: ignore[no-untyped-def]
            if league.key == "LIGA_MX":
                raise DataFetchError("caída simulada")
            return sim.fetch_matches(league, season)

    flaky = Flaky()
    slate = analyze_slate(DataFetcher(flaky, odds_provider=sim), ["EPL", "LIGA_MX"], PipelineConfig(use_ml=False))
    assert "LIGA_MX" in slate.errors and "EPL" in slate.results
    with pytest.raises(DataFetchError):
        analyze_slate(DataFetcher(flaky, odds_provider=sim), ["LIGA_MX"], PipelineConfig(use_ml=False))
