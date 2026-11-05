"""Tests del ajuste por lesiones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictive_odds.data_fetcher import DataFetcher, SimulatedDataProvider
from predictive_odds.domain import Selection
from predictive_odds.injuries import InjuryAdjuster, InjuryImpactConfig, PlayerImpact
from predictive_odds.pipeline import AnalysisPipeline, PipelineConfig
from predictive_odds.statistical_models import (
    EnsemblePredictor,
    FeatureEngineer,
    MLOutcomeModel,
    PoissonModel,
    TeamAdjustment,
)


def _injuries(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["team", "player", "type"])


def test_adjuster_weights_caps_and_aliases() -> None:
    adj = InjuryAdjuster().adjustments(_injuries([
        ("Man Utd", "A", "Missing Fixture"), ("Man Utd", "B", "Questionable"), ("Man Utd", "A", "Missing Fixture"),
        ("Real Madrid", "C", "Missing Fixture"),
    ]), teams=["Manchester United", "Chelsea"])
    assert set(adj) == {"Manchester United"}  # alias resuelto, equipo desconocido ignorado
    a = adj["Manchester United"]
    assert a.attack == pytest.approx(1 - 0.03 * 1.5)  # duplicado ignorado, duda a medias
    assert a.defense == pytest.approx(1 + 0.02 * 1.5)
    many = InjuryAdjuster().adjustments(_injuries([("X", f"p{i}", "Missing Fixture") for i in range(20)]))
    assert many["X"].attack == pytest.approx(1 - 0.15)
    star = InjuryAdjuster(InjuryImpactConfig(player_impacts={"Star": PlayerImpact(0.2, 0.0)}))
    # El impacto específico sustituye a la heurística, pero el tope total sigue aplicando
    assert star.adjustments(_injuries([("X", "Star", "Missing Fixture")]))["X"].attack == pytest.approx(0.85)
    assert InjuryAdjuster().adjustments(pd.DataFrame()) == {}
    with pytest.raises(ValueError):
        TeamAdjustment(attack=0.0)


def test_adjustment_changes_lambdas_and_keeps_coherence() -> None:
    sim = SimulatedDataProvider()
    hist = DataFetcher(sim).get_history("EPL", [2024, 2025])
    poisson = PoissonModel().fit(hist)
    fe = FeatureEngineer()
    ml = MLOutcomeModel().fit(fe.fit_transform(hist))
    ens = EnsemblePredictor(poisson, ml, fe)
    base = ens.predict("Arsenal", "Chelsea")
    hurt = ens.predict("Arsenal", "Chelsea", home_adj=TeamAdjustment(attack=0.85, defense=1.1))
    assert hurt.lambda_home == pytest.approx(base.lambda_home * 0.85)
    assert hurt.lambda_away == pytest.approx(base.lambda_away * 1.1)
    assert hurt.probability(Selection.home()) < base.probability(Selection.home())
    assert hurt.sources["ml"][Selection.home()] < base.sources["ml"][Selection.home()]
    total = sum(hurt.probability(s) for s in (Selection.home(), Selection.draw(), Selection.away()))
    assert total == pytest.approx(1.0)


def test_pipeline_applies_simulated_injuries() -> None:
    sim = SimulatedDataProvider()
    fetcher = DataFetcher(sim, odds_provider=sim)
    on = AnalysisPipeline(fetcher, PipelineConfig(use_ml=False)).run(evaluate=False)
    off = AnalysisPipeline(fetcher, PipelineConfig(use_ml=False, use_injuries=False)).run(evaluate=False)
    assert on.adjustments and not off.adjustments
    injured = next(iter(on.adjustments))
    mid, pred = next((m, p) for m, p in on.predictions.items() if injured in (p.home_team, p.away_team))
    lam_on = pred.lambda_home if pred.home_team == injured else pred.lambda_away
    lam_off = off.predictions[mid].lambda_home if pred.home_team == injured else off.predictions[mid].lambda_away
    assert lam_on < lam_off
    assert np.isfinite(lam_on)
