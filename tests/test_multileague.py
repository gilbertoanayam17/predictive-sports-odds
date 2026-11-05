"""Tests del modelo de Poisson multiliga (UCL)."""

from __future__ import annotations

import pandas as pd
import pytest

from predictive_odds.config import ConfigurationError, get_league
from predictive_odds.data_fetcher import DataFetcher, SimulatedDataProvider, SimulationConfig
from predictive_odds.pipeline import AnalysisPipeline, PipelineConfig
from predictive_odds.statistical_models import PoissonModel, infer_team_leagues


def test_infer_team_leagues() -> None:
    df = pd.DataFrame({"home_team": ["A", "B", "A", "C"], "away_team": ["B", "A", "C", "A"],
                       "league": ["EPL", "EPL", "UCL", "UCL"]})
    assert infer_team_leagues(df, ["UCL"]) == {"A": "EPL", "B": "EPL", "C": "OTHER"}


def test_multileague_recovers_league_gap() -> None:
    sim = SimulatedDataProvider(SimulationConfig(seed=2, league_offsets={"EPL": 0.4, "OTHER": 0.0}))
    fetch = DataFetcher(sim)
    ucl, epl = fetch.get_history("UCL", [2024, 2025]), fetch.get_history("EPL", [2024, 2025])
    combined = pd.concat([ucl, epl], ignore_index=True)
    model = PoissonModel().fit(combined, team_leagues=infer_team_leagues(combined, ["UCL"]))
    assert model.league_strength["EPL"] - model.league_strength["OTHER"] > 0.15
    # Un equipo nuevo de la EPL hereda la fuerza media de su liga
    model.team_leagues["Equipo Nuevo"] = "EPL"
    assert model.expected_goals("Equipo Nuevo", "Real Madrid")[0] > model.expected_goals("Otro", "Real Madrid")[0]


def test_pipeline_ucl_with_extra_leagues() -> None:
    sim = SimulatedDataProvider()
    res = AnalysisPipeline(DataFetcher(sim, odds_provider=sim),
                           PipelineConfig(league_key="UCL", use_ml=False, extra_leagues=("EPL", "NOPE_LEAGUE")))
    with pytest.raises(ConfigurationError):
        res.run(evaluate=False)  # liga inexistente -> error de configuración, no silencioso
    ok = AnalysisPipeline(DataFetcher(sim, odds_provider=sim),
                          PipelineConfig(league_key="UCL", use_ml=False, extra_leagues=("EPL",))).run(evaluate=False)
    assert ok.predictions
    assert set(ok.team_ratings["team"]) <= set(sim.fetch_matches(get_league("UCL"), 2025)["home_team"])
