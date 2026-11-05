"""Tests de calibración y persistencia de modelos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictive_odds.calibration import (
    ProbabilityCalibrator,
    calibration_report,
    expected_calibration_error,
    reliability_table,
)
from predictive_odds.data_fetcher import DataFetcher, SimulatedDataProvider
from predictive_odds.domain import Selection
from predictive_odds.model_store import BundleMetadata, ModelBundle, ModelStoreError, load_bundle, save_bundle
from predictive_odds.pipeline import AnalysisPipeline, PipelineConfig


def _miscalibrated(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, n)
    y = (rng.uniform(size=n) < true_p).astype(float)
    reported = np.clip(true_p + 0.6 * (true_p - 0.5), 0.01, 0.99)  # sobreconfiado
    return reported, y


def test_ece_and_reliability() -> None:
    assert expected_calibration_error([0.0, 1.0], [0, 1]) == 0.0
    p, y = _miscalibrated()
    assert expected_calibration_error(p, y) > 0.05
    table = reliability_table(p, y)
    assert table["n"].sum() == len(p)
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [1, 0])


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibrator_fixes_overconfidence(method: str) -> None:
    p, y = _miscalibrated()
    markets = np.array(["OU"] * len(p))
    cal = ProbabilityCalibrator(method).fit(p[:2000], y[:2000], markets[:2000])  # type: ignore[arg-type]
    fixed = cal.transform(p[2000:], "OU")
    assert expected_calibration_error(fixed, y[2000:]) < expected_calibration_error(p[2000:], y[2000:]) / 2
    assert np.array_equal(cal.transform(p[:5], "1X2"), p[:5])  # mercado no calibrado: sin cambios


def test_calibration_report_and_prediction() -> None:
    p, y = _miscalibrated()
    ev = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=len(p), freq="h"),
                       "market": "OU", "model_prob": p, "won": y.astype(bool)})
    rep = calibration_report(ev)
    assert rep["ece_despues"] < rep["ece_antes"]

    sim = SimulatedDataProvider()
    result = AnalysisPipeline(DataFetcher(sim, odds_provider=sim), PipelineConfig(use_ml=False)).run(evaluate=False)
    pred = next(iter(result.predictions.values()))
    cal = ProbabilityCalibrator().fit(p, y, np.array(["OU"] * len(p)))
    cal.calibrate_prediction(pred)
    assert pred.probability(Selection.over(2.5)) + pred.probability(Selection.under(2.5)) == pytest.approx(1.0)
    assert "calibrated" in pred.sources


def test_bundle_roundtrip_and_errors(tmp_path: Path) -> None:
    sim = SimulatedDataProvider()
    fetcher = DataFetcher(sim, odds_provider=sim)
    path = tmp_path / "epl.joblib"
    cfg = PipelineConfig(use_ml=False, model_path=str(path))
    first = AnalysisPipeline(fetcher, cfg).run(evaluate=False)
    assert path.exists() and path.with_suffix(".joblib.json").exists()
    bundle = load_bundle(path, expected_league="EPL")
    assert isinstance(bundle, ModelBundle) and bundle.metadata.n_matches > 0
    second = AnalysisPipeline(fetcher, cfg).run(evaluate=False)  # reutiliza el bundle
    mid = next(iter(first.predictions))
    assert second.predictions[mid].lambda_home == pytest.approx(first.predictions[mid].lambda_home)

    with pytest.raises(ModelStoreError):
        load_bundle(path, expected_league="LIGA_MX")
    with pytest.raises(ModelStoreError):
        load_bundle(tmp_path / "missing.joblib")
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"not a pickle")
    with pytest.raises(ModelStoreError):
        load_bundle(corrupt)
    assert bundle.is_stale(pd.Timestamp(bundle.metadata.trained_until) + pd.Timedelta(days=30))
    assert not bundle.is_stale(pd.Timestamp(bundle.metadata.trained_until))
    _ = BundleMetadata  # exportado
    _ = save_bundle
