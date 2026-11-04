"""Tests del estudio por liga y de la aplicación automática de perfiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from predictive_odds.data_fetcher import DataFetcher, SimulatedDataProvider, SimulationConfig
from predictive_odds.league_study import (
    LeagueProfile,
    StudyConfig,
    _market_stats,
    allowed_markets_for,
    load_profiles,
    run_study,
    save_profiles,
    study_frame,
)
from predictive_odds.pipeline import AnalysisPipeline, PipelineConfig


def _bets(market: str, n: int, clv: float, pnl: float) -> pd.DataFrame:
    return pd.DataFrame({"type": "single", "market": market, "stake": 10.0, "pnl": pnl, "clv": clv},
                        index=range(n))


def test_market_verdicts() -> None:
    bets = pd.concat([_bets("AH", 100, 0.03, -1.0), _bets("BTTS", 100, -0.02, 2.0), _bets("1X2", 20, 0.1, 5.0)],
                     ignore_index=True)
    stats = {m.market: m for m in _market_stats(bets, min_bets=80, clv_comparable=True)}
    assert stats["AH"].recommended  # CLV positivo aunque el ROI sea negativo: manda el CLV
    assert not stats["BTTS"].recommended  # ROI positivo pero CLV negativo: suerte
    assert not stats["1X2"].recommended and "insuficiente" in stats["1X2"].reason
    extra = {m.market: m for m in _market_stats(bets, min_bets=40, clv_comparable=False)}
    assert extra["BTTS"].recommended and "CLV no comparable" in extra["BTTS"].reason
    assert not extra["AH"].recommended


def test_study_save_load_and_apply(tmp_path: Path) -> None:
    sim = SimulatedDataProvider(SimulationConfig(seed=2, seasons=(2024, 2025)))
    profiles, errors = run_study(sim, ["EPL", "NO_EXISTE"], StudyConfig(seasons_back=1, min_bets=30), season=2025)
    assert "NO_EXISTE" in errors and profiles["EPL"].seasons == [2024, 2025]
    assert not study_frame(profiles).empty
    path = tmp_path / "profiles.json"
    save_profiles(profiles, path)
    save_profiles({"LIGA_MX": LeagueProfile("LIGA_MX", [2025], [], False)}, path)  # fusiona, no pisa
    loaded = load_profiles(path)
    assert set(loaded) == {"EPL", "LIGA_MX"} and loaded["EPL"].stats
    assert allowed_markets_for("LIGA_MX", path) == () and allowed_markets_for("SERIE_A", path) is None

    only_ou = tmp_path / "ou.json"
    save_profiles({"EPL": LeagueProfile("EPL", [2025], ["OU"], True)}, only_ou)
    fetch = DataFetcher(sim, odds_provider=sim)
    r = AnalysisPipeline(fetch, PipelineConfig(seasons=(2024, 2025), use_ml=False, profile_path=str(only_ou))).run(
        evaluate=False)
    assert r.profile_markets == ("OU",) and r.value_bets
    assert {p.selection.market.value for p in r.value_bets} == {"OU"}
    assert all(leg.selection.market.value == "OU" for par in r.parlays for leg in par.legs)
    none = AnalysisPipeline(fetch, PipelineConfig(seasons=(2024, 2025), use_ml=False,
                                                  profile_path=str(path))).run(evaluate=False)
    assert none.profile_markets is not None


def test_corrupt_profiles_file(tmp_path: Path) -> None:
    bad = tmp_path / "p.json"
    bad.write_text("{no es json", encoding="utf-8")
    assert load_profiles(bad) == {} and allowed_markets_for("EPL", bad) is None
