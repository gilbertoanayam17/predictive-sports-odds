"""Tests del ajuste de hiperparámetros."""

from __future__ import annotations

import pandas as pd
import pytest

from predictive_odds.data_fetcher import DataFetcher, SimulatedDataProvider
from predictive_odds.statistical_models import InsufficientDataError, PoissonModelConfig
from predictive_odds.tuning import evaluate_poisson_config, rolling_origin_splits, tune_ml_weight, tune_poisson


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return DataFetcher(SimulatedDataProvider()).get_history("EPL", [2024, 2025])


def test_splits_are_chronological(history: pd.DataFrame) -> None:
    splits = rolling_origin_splits(history, n_folds=4)
    assert len(splits) == 4
    assert all(a.cutoff < b.cutoff for a, b in zip(splits, splits[1:]))
    assert (history["date"] < splits[0].cutoff).sum() >= 150
    with pytest.raises(InsufficientDataError):
        rolling_origin_splits(history.head(100))


def test_tune_poisson_ranks_and_validates(history: pd.DataFrame) -> None:
    splits = rolling_origin_splits(history, n_folds=3)
    res = tune_poisson(history, {"xg_weight": [0.0, 0.4]}, splits=splits)
    assert list(res.table["score"]) == sorted(res.table["score"])
    assert res.best["xg_weight"] in (0.0, 0.4)
    default = evaluate_poisson_config(history, PoissonModelConfig(), splits)["score"]
    assert res.baseline_score == pytest.approx(default)
    with pytest.raises(ValueError):
        tune_poisson(history, {"no_existe": [1]}, splits=splits)


def test_tune_ml_weight(history: pd.DataFrame) -> None:
    res = tune_ml_weight(history, [0.0, 0.5])
    assert set(res.table["ml_weight"]) == {0.0, 0.5}
    assert res.table["n"].iloc[0] > 0
