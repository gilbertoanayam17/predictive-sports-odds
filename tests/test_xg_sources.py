"""Tests de fuentes de xG (sin red)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from predictive_odds.config import get_league
from predictive_odds.data_fetcher import (
    DataFetcher,
    DataValidationError,
    HttpClient,
    ProviderNotSupportedError,
    SimulatedDataProvider,
)
from predictive_odds.xg_sources import UnderstatProvider, XgEnrichedProvider, merge_xg, parse_understat_html, understat_to_matches

ITEMS = [
    {"id": "1", "isResult": True, "h": {"title": "Manchester United"}, "a": {"title": "Arsenal"},
     "goals": {"h": "1", "a": "2"}, "xG": {"h": "0.85", "a": "1.94"}, "datetime": "2025-08-17 16:30:00"},
    {"id": "2", "isResult": False, "h": {"title": "Arsenal"}, "a": {"title": "Leeds"},
     "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None}, "datetime": "2026-05-01 15:00:00"},
    {"id": "3", "broken": True},
]


def _html(items: list[dict[str, Any]]) -> str:
    """Página con el formato documentado de Understat (JSON escapado en hexadecimal)."""
    escaped = "".join(f"\\x{ord(c):02x}" if c in "\"'{}[]:," else c for c in json.dumps(items))
    return f"<script>var datesData = JSON.parse('{escaped}');</script>"


def test_parse_understat_html_and_convert() -> None:
    items = parse_understat_html(_html(ITEMS))
    df = understat_to_matches(items, "EPL", 2025)
    assert list(df["status"]) == ["FT", "NS"]  # el mal formado se omite
    assert df.iloc[0]["home_xg"] == pytest.approx(0.85)
    with pytest.raises(DataValidationError):
        parse_understat_html("<html>sin datos</html>")


class _FakeClient(HttpClient):
    def __init__(self, html: str, payload: Any) -> None:
        super().__init__("https://x")
        self.html, self.payload = html, payload

    def get_text(self, path: str, params: Any = None, use_cache: bool = True) -> str:
        return self.html

    def get_json(self, path: str, params: Any = None, use_cache: bool = True) -> Any:
        return self.payload


def test_understat_provider_fallback_and_errors() -> None:
    epl = get_league("EPL")
    assert len(UnderstatProvider(_FakeClient(_html(ITEMS), None)).fetch_matches(epl, 2025)) == 2
    fallback = UnderstatProvider(_FakeClient("<html/>", {"dates": ITEMS}))
    assert len(fallback.fetch_matches(epl, 2025)) == 2
    with pytest.raises(DataValidationError):
        UnderstatProvider(_FakeClient("<html/>", {"otra": 1})).fetch_matches(epl, 2025)
    with pytest.raises(ProviderNotSupportedError):
        UnderstatProvider(_FakeClient("", None)).fetch_matches(get_league("LIGA_MX"), 2025)


def test_merge_xg_matches_names_dates_and_scores() -> None:
    sim = SimulatedDataProvider()
    full = DataFetcher(sim).get_history("EPL", [2025])
    target = full.assign(home_xg=np.nan, away_xg=np.nan)
    # Fuente externa: otros nombres, fecha desplazada 1 día y un marcador alterado
    src = full.copy()
    src["home_team"] = src["home_team"].replace({"Manchester United": "Man Utd"})
    src["date"] = src["date"] + pd.Timedelta(days=1)
    src.loc[src.index[0], "home_goals"] = 99
    merged = merge_xg(target, src)
    assert merged["home_xg"].notna().sum() == len(full) - 1  # el de marcador distinto no se fusiona
    np.testing.assert_allclose(merged["home_xg"].iloc[1:], full["home_xg"].iloc[1:])
    # Sin overwrite no toca el xG existente
    again = merge_xg(merged.assign(home_xg=merged["home_xg"].fillna(-1)), src)
    assert (again["home_xg"] == -1).sum() == 1


def test_xg_enriched_provider_tolerates_failures() -> None:
    sim = SimulatedDataProvider()
    epl = get_league("EPL")
    no_xg = sim.fetch_matches(epl, 2025).assign(home_xg=np.nan, away_xg=np.nan)

    class Base(SimulatedDataProvider):
        def fetch_matches(self, league: Any, season: int) -> pd.DataFrame:
            return no_xg.copy()

    enriched = XgEnrichedProvider(Base(), sim).fetch_matches(epl, 2025)
    assert enriched["home_xg"].notna().sum() == no_xg["status"].eq("FT").sum()
    broken = XgEnrichedProvider(Base(), UnderstatProvider(_FakeClient("<html/>", {})))
    assert broken.fetch_matches(epl, 2025)["home_xg"].isna().all()  # falla la fuente de xG → sin xG, sin error
