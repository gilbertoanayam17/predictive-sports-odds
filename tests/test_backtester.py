"""Tests del backtester walk-forward."""

from __future__ import annotations

import pandas as pd
import pytest

from predictive_odds.backtester import BacktestConfig, Backtester, selection_wins
from predictive_odds.config import get_league
from predictive_odds.data_fetcher import CLOSING_BOOKMAKER, DataFetcher, SimulatedDataProvider, SimulationConfig
from predictive_odds.domain import Selection
from predictive_odds.statistical_models import InsufficientDataError


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    sim = SimulatedDataProvider(SimulationConfig(seed=5, seasons=(2025,), played_rounds_current=24))
    hist = DataFetcher(sim).get_history("EPL", [2025])
    odds = sim.fetch_historical_odds(get_league("EPL"), 2025)
    return hist, odds


def test_selection_wins() -> None:
    assert selection_wins(Selection.home(), 2, 1)
    assert not selection_wins(Selection.over(2.5), 1, 1)
    assert selection_wins(Selection.btts(True), 12, 1)  # fuera de la rejilla por defecto


def test_historical_odds_include_closing(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    hist, odds = data
    assert CLOSING_BOOKMAKER in set(odds["bookmaker"])
    assert set(odds["match_id"]) == set(hist["match_id"])


def test_backtest_runs_and_is_consistent(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    hist, odds = data
    res = Backtester(hist, odds, BacktestConfig(use_ml=False, warmup_matches=120, include_parlays=True)).run()
    m = res.metrics
    assert m["apuestas"] == len(res.bets) > 0
    assert m["bankroll_final"] == pytest.approx(m["bankroll_inicial"] + res.bets["pnl"].sum())
    assert (res.bets["date"] >= hist["date"].iloc[120]).all()  # no apuesta durante el calentamiento
    assert 0 <= m["max_drawdown"] < 1
    assert res.bets["clv"].notna().all()
    cal = res.calibration()
    assert cal["n"].sum() == len(res.evaluated)


def test_backtest_requires_enough_data(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    hist, odds = data
    with pytest.raises(InsufficientDataError):
        Backtester(hist.head(50), odds, BacktestConfig(warmup_matches=100))
