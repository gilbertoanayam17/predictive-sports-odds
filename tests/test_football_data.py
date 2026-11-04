"""Tests del lector de football-data.co.uk con CSV en los formatos publicados (sin red)."""

from __future__ import annotations

from typing import Any

import pytest

from predictive_odds.config import get_league
from predictive_odds.data_fetcher import (
    CLOSING_BOOKMAKER,
    DataValidationError,
    FootballDataCoUkProvider,
    HttpClient,
    ProviderNotSupportedError,
)

MAIN_CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,PSH,PSD,PSA,AvgH,AvgD,AvgA,B365>2.5,B365<2.5,AHh,B365AHH,B365AHA,PSCH,PSCD,PSCA,PC>2.5,PC<2.5,AHCh,PCAHH,PCAHA
E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,1.30,6.00,9.50,1.31,6.20,10.0,1.30,6.00,9.40,1.44,2.75,-1.75,1.95,1.95,1.28,6.40,11.0,1.40,3.00,-1.75,1.93,1.97
E0,16/08/2025,12:30,Aston Villa,Newcastle,0,0,D,2.30,3.50,3.00,2.35,3.55,3.05,2.32,3.48,3.02,1.80,2.00,-0.25,2.05,1.85,2.40,3.50,2.95,1.85,2.02,0,1.80,2.10
,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
"""

EXTRA_CSV = """Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA,B365CH,B365CD,B365CA
Mexico,Liga MX,2024/2025,20/04/2025,02:00,Toluca,Pachuca,2,1,H,1.70,3.90,4.80,1.75,4.00,5.00,1.68,3.85,4.70,1.67,3.80,4.75
Mexico,Liga MX,2025/2026,11/07/2025,02:00,Puebla,Queretaro,1,1,D,2.20,3.30,3.40,2.25,3.40,3.50,2.15,3.25,3.35,2.10,3.30,3.40
"""


class _Client(HttpClient):
    def __init__(self, text: str) -> None:
        super().__init__("https://x")
        self.text, self.paths = text, []

    def get_text(self, path: str, params: Any = None, use_cache: bool = True) -> str:
        self.paths.append(path)
        return self.text


def test_main_format_results_and_odds() -> None:
    client = _Client(MAIN_CSV)
    fd = FootballDataCoUkProvider(client)
    epl = get_league("EPL")
    matches = fd.fetch_matches(epl, 2025)
    assert client.paths[0] == "mmz4281/2526/E0.csv"
    assert len(matches) == 2  # la fila vacía se descarta
    assert matches.iloc[0]["date"].day == 15 and matches.iloc[0]["date"].month == 8  # dd/mm
    odds = fd.fetch_historical_odds(epl, 2025)
    books = set(odds["bookmaker"])
    assert {"bet365", "pinnacle", "average", CLOSING_BOOKMAKER} <= books
    ah = odds[(odds["market"] == "AH") & (odds["bookmaker"] == "bet365")]
    assert set(zip(ah["selection"], ah["line"])) >= {("HOME", -1.75), ("AWAY", 1.75), ("HOME", -0.25)}
    closing_ah = odds[(odds["market"] == "AH") & (odds["bookmaker"] == CLOSING_BOOKMAKER)]
    assert ("HOME", 0.0) in set(zip(closing_ah["selection"], closing_ah["line"]))
    no_close = fd.fetch_historical_odds(epl, 2025, include_closing=False)
    assert CLOSING_BOOKMAKER not in set(no_close["bookmaker"])


def test_extra_format_uses_closing_prices_and_filters_season() -> None:
    client = _Client(EXTRA_CSV)
    fd = FootballDataCoUkProvider(client)
    mx = get_league("LIGA_MX")
    matches = fd.fetch_matches(mx, 2025)
    assert client.paths[0] == "new/MEX.csv"
    assert list(matches["home_team"]) == ["Puebla"]  # solo la temporada 2025/2026
    odds = fd.fetch_historical_odds(mx, 2025)
    # Sin precios pre-partido: se usan los de cierre de cada casa como precio de apuesta
    assert {"bet365", "average"} <= set(odds["bookmaker"])
    assert "max" not in set(odds["bookmaker"])  # el máximo del mercado no es una casa real
    b365 = odds[(odds["bookmaker"] == "bet365") & (odds["selection"] == "HOME")]
    assert b365["odds"].iloc[0] == 2.10


def test_bad_csv_and_unsupported_league() -> None:
    with pytest.raises(DataValidationError):
        FootballDataCoUkProvider(_Client("a,b\n1,2\n")).fetch_matches(get_league("EPL"), 2025)
    with pytest.raises(ProviderNotSupportedError):
        FootballDataCoUkProvider(_Client("")).fetch_matches(get_league("UCL"), 2025)
