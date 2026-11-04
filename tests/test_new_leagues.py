"""La Liga, Serie A y Bundesliga + error de plan de API-Football."""

from __future__ import annotations

from typing import Any

import pytest

from predictive_odds.config import get_league, selectable_leagues
from predictive_odds.data_fetcher import (
    ApiFootballProvider,
    DataFetcher,
    HttpClient,
    ProviderNotSupportedError,
    SimulatedDataProvider,
)
from predictive_odds.pipeline import AnalysisPipeline, PipelineConfig, analyze_slate


def test_selectable_leagues_exclude_champions() -> None:
    leagues = selectable_leagues()
    assert {"EPL", "LA_LIGA", "SERIE_A", "BUNDESLIGA", "LIGA_MX"} <= set(leagues)
    assert "UCL" not in leagues
    for key in ("LA_LIGA", "SERIE_A", "BUNDESLIGA"):
        cfg = get_league(key)
        assert cfg.football_data_code and cfg.football_data_format == "main" and cfg.odds_api_sport_key


@pytest.mark.parametrize("key,teams", [("LA_LIGA", 20), ("SERIE_A", 20), ("BUNDESLIGA", 18)])
def test_new_leagues_run_end_to_end(key: str, teams: int) -> None:
    sim = SimulatedDataProvider()
    r = AnalysisPipeline(DataFetcher(sim, odds_provider=sim),
                         PipelineConfig(league_key=key, use_ml=False)).run(evaluate=False)
    assert len(r.predictions) == teams // 2 and len(r.standings) == teams and not r.unknown_teams


def test_slate_with_european_leagues() -> None:
    sim = SimulatedDataProvider()
    slate = analyze_slate(DataFetcher(sim, odds_provider=sim), ["EPL", "LA_LIGA", "SERIE_A", "BUNDESLIGA"],
                          PipelineConfig(use_ml=False, max_singles=10))
    assert set(slate.results) == {"EPL", "LA_LIGA", "SERIE_A", "BUNDESLIGA"} and not slate.errors


class _Client(HttpClient):
    def __init__(self, payload: Any) -> None:
        super().__init__("https://x")
        self.payload = payload

    def get_json(self, path: str, params: Any = None, use_cache: bool = True) -> Any:
        return self.payload


def test_api_football_free_plan_error_is_explained() -> None:
    payload = {"errors": {"plan": "Free plans do not have access to this season, try from 2022 to 2024."},
               "response": []}
    with pytest.raises(ProviderNotSupportedError, match="plan gratuito"):
        ApiFootballProvider("k", client=_Client(payload)).fetch_matches(get_league("EPL"), 2026)
