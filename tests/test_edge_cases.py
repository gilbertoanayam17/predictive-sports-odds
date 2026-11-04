"""Regresiones de casos límite del pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from predictive_odds.config import get_league
from predictive_odds.data_fetcher import DataFetcher, DataFetchError, SimulatedDataProvider, SimulationConfig, validate_odds
from predictive_odds.pipeline import AnalysisPipeline, PipelineConfig
from predictive_odds.statistical_models import InsufficientDataError


def test_without_odds_provider_only_predicts() -> None:
    r = AnalysisPipeline(DataFetcher(SimulatedDataProvider()), PipelineConfig(use_ml=False)).run(evaluate=False)
    assert r.predictions and not r.value_bets and not r.parlays and r.total_stake == 0


def test_finished_season_raises_clear_error() -> None:
    sim = SimulatedDataProvider(SimulationConfig(played_rounds_current=38))
    with pytest.raises(DataFetchError, match="partidos próximos"):
        AnalysisPipeline(DataFetcher(sim, odds_provider=sim), PipelineConfig(use_ml=False)).run(evaluate=False)


def test_corrupt_odds_rows_are_dropped() -> None:
    odds = SimulatedDataProvider().fetch_odds(get_league("EPL"))
    bad = pd.DataFrame([{"match_id": "x", "league": "EPL", "home_team": "A", "away_team": "B", "commence_time": None,
                         "bookmaker": "z", "market": "1X2", "selection": "HOME", "line": None, "odds": "abc"}])
    assert len(validate_odds(pd.concat([odds, bad]))) == len(odds)


def test_early_season_disables_ml_and_evaluation() -> None:
    sim = SimulatedDataProvider(SimulationConfig(seasons=(2025,), played_rounds_current=4))
    r = AnalysisPipeline(DataFetcher(sim, odds_provider=sim), PipelineConfig(seasons=(2025,))).run()
    assert r.predictions and r.evaluation == {}
    assert all("ml" not in p.sources for p in r.predictions.values())


def test_too_little_history_raises() -> None:
    sim = SimulatedDataProvider(SimulationConfig(seasons=(2025,), played_rounds_current=2))
    with pytest.raises(InsufficientDataError):
        AnalysisPipeline(DataFetcher(sim, odds_provider=sim), PipelineConfig(seasons=(2025,))).run()
