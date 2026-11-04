"""Regresiones de la revisión de código (fase 5, tarea 25)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from predictive_odds.config import get_league
from predictive_odds.data_fetcher import (
    ApiFootballProvider,
    DataFetcher,
    HttpClient,
    ProviderNotSupportedError,
    ResultsProvider,
    SimulatedDataProvider,
)
from predictive_odds.domain import Selection
from predictive_odds.league_study import run_study
from predictive_odds.notifier import format_alert
from predictive_odds.parlay_builder import Pick, ValueBetFinder
from predictive_odds.pipeline import AnalysisPipeline, PipelineConfig
from predictive_odds.settings import env_file_keys, load_env


def test_env_keys_visible_on_every_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PREDICTIVE_ODDS_NO_DOTENV")
    env = tmp_path / ".env"
    env.write_text("ODDS_API_KEY=abc\n", encoding="utf-8")
    assert load_env(env) == ["ODDS_API_KEY"]
    assert load_env(env) == []  # segunda recarga: ya estaba cargada
    assert env_file_keys(env) == ["ODDS_API_KEY"]  # pero se sigue informando
    monkeypatch.setenv("ODDS_API_KEY", "otra_del_sistema")
    assert env_file_keys(env) == []  # la del sistema manda: no viene del .env
    assert os.environ["ODDS_API_KEY"] == "otra_del_sistema"


def test_alert_with_missing_kickoff() -> None:
    pick = Pick("m", "EPL", "A", "B", pd.NaT, Selection.home(), 2.0, "b", 0.6, 0.5, 0.01, decision_prob=0.6)
    assert "A vs B" in format_alert([pick], 1000)


class _PlanClient(HttpClient):
    def __init__(self) -> None:
        super().__init__("https://x")
        self.calls = 0

    def get_json(self, path: str, params: Any = None, use_cache: bool = True) -> Any:
        self.calls += 1
        return {"errors": {"plan": "Free plans do not have access to this season"}, "response": []}


def test_api_football_plan_error_not_repeated() -> None:
    client = _PlanClient()
    for _ in range(3):
        with pytest.raises(ProviderNotSupportedError):
            ApiFootballProvider("k", client=client).fetch_matches(get_league("EPL"), 2026)
    assert client.calls == 1


def test_consensus_handles_lines_without_value() -> None:
    odds = SimulatedDataProvider().fetch_odds(get_league("EPL"))
    fair = ValueBetFinder().market_consensus(odds)
    mid = odds["match_id"].iloc[0]
    assert sum(fair[(mid, s)] for s in (Selection.home(), Selection.draw(), Selection.away())) == pytest.approx(1)
    assert (mid, Selection.btts(True)) in fair and (mid, Selection.over(2.5)) in fair


def test_model_bundle_refits_when_config_changes(tmp_path: Path) -> None:
    sim = SimulatedDataProvider()
    fetch = DataFetcher(sim, odds_provider=sim)
    path = str(tmp_path / "m.joblib")
    base = PipelineConfig(use_ml=False, model_path=path)
    first = AnalysisPipeline(fetch, base)
    fp1 = first.model_fingerprint()
    first.run(evaluate=False)
    changed = AnalysisPipeline(fetch, PipelineConfig(use_ml=False, model_path=path, ml_weight=0.1))
    assert changed.model_fingerprint() != fp1
    from predictive_odds.model_store import load_bundle

    assert load_bundle(path).metadata.extra["config"] == fp1
    changed.run(evaluate=False)
    assert load_bundle(path).metadata.extra["config"] == changed.model_fingerprint()  # se reentrenó


def test_study_rejects_provider_without_history() -> None:
    class NoOdds(ResultsProvider):
        name = "sin-historico"

        def fetch_matches(self, league: Any, season: int) -> pd.DataFrame:
            return SimulatedDataProvider().fetch_matches(league, 2025)

    profiles, errors = run_study(NoOdds(), ["EPL"], season=2025)
    assert not profiles and "momios históricos" in errors["EPL"]
